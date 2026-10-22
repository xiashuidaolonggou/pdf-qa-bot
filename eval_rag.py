"""
eval_rag.py — RAG 知识库问答的检索与问答评测脚本

默认指标：
    1. 命中率 Recall@k        —— top-k 片段里是否有包含"关键证据短语"的片段
    2. MRR@k                 —— 第一个命中片段的排名倒数均值（衡量排得靠不靠前）
    3. 检索耗时 / 生成耗时     —— 均值、中位数、P95
    4. 拒答正确率             —— 文档里确实没有答案的问题，模型是否明确说"没有"
    5. 无依据回答率（幻觉率估计）—— LLM 判定回答不被检索片段支持的比例

用法（在 E:\\shixi\\pdf-qa-bot 目录下，先激活 venv）：
    python eval_rag.py --chunks                       # 只统计知识库规模
    python eval_rag.py --run                          # 跑评测：MMR(k=6) + LLM 判定
    python eval_rag.py --run --no-judge               # 只测检索与耗时，不调用生成（省额度）
    python eval_rag.py --run --mode similarity        # 用纯相似度检索跑一遍做对比
    python eval_rag.py --run --k 3 --k 6 --k 9        # 扫多个 k
    python eval_rag.py --gen 20                       # 从知识库里自动生成候选测试集

输出：
    eval_report.md    —— 可直接贴进 README / 简历附件的评测报告
    eval_results.json —— 明细数据（每条问题的命中、耗时、判定）

测试集格式（testset.json，数组）：
    {"id": "q1", "type": "answerable", "question": "...", "key_phrase": "答案里必定出现的原文片段"}
    {"id": "q2", "type": "unanswerable", "question": "文档里根本不存在答案的问题"}
说明：key_phrase 要写"原文里逐字出现的一段话"，脚本会做去空白/去标点匹配。
"""
import argparse
import json
import os
import re
import statistics
import sys
import time
from typing import Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8")

import config
import rag_engine

TESTSET_PATH = os.path.join(config.BASE_DIR, "testset.json")
REPORT_PATH = os.path.join(config.BASE_DIR, "eval_report.md")
RESULT_PATH = os.path.join(config.BASE_DIR, "eval_results.json")

_PUNCT = re.compile(r"[\s\u3000,，。.、;；:：!！?？\"'“”‘’()（）\[\]【】<>《》\-—|/\\]+")


def norm(text: str) -> str:
    """去掉空白和标点，用于宽松匹配"""
    return _PUNCT.sub("", (text or "").lower())


# ---------------------------------------------------------------- 网络 / 代理
GOOGLE_HOST = "generativelanguage.googleapis.com"
PROXY_PORTS = [7890, 7891, 7897, 7898, 10808, 10809, 2080, 1080, 3065, 3066, 3067, 3064]


def setup_proxy(explicit: Optional[str]) -> Optional[str]:
    """优先用 --proxy；其次用环境变量；最后自动探测本地常见代理端口"""
    if explicit:
        os.environ["HTTPS_PROXY"] = explicit
        os.environ["HTTP_PROXY"] = explicit
        return explicit
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        if os.environ.get(name):
            return os.environ[name]
    import socket

    for port in PROXY_PORTS:
        s = socket.socket()
        s.settimeout(0.3)
        try:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                url = f"http://127.0.0.1:{port}"
                os.environ["HTTPS_PROXY"] = url
                os.environ["HTTP_PROXY"] = url
                print(f"检测到本地代理 {url}，已自动启用", flush=True)
                return url
        finally:
            s.close()
    return None


def check_network(proxy: Optional[str]) -> bool:
    """确认能连上 Gemini 接口；失败时给出可执行的修复建议，而不是一直卡住"""
    import socket
    from urllib.parse import urlparse

    if proxy:
        try:
            p = urlparse(proxy)
            s = socket.create_connection((p.hostname or "127.0.0.1", p.port or 80), timeout=5)
            s.sendall(f"CONNECT {GOOGLE_HOST}:443 HTTP/1.1\r\nHost: {GOOGLE_HOST}:443\r\n\r\n".encode())
            resp = s.recv(64).decode(errors="ignore")
            s.close()
            return "200" in resp.split("\r\n")[0]
        except Exception:
            return False
    try:
        socket.create_connection((GOOGLE_HOST, 443), timeout=5).close()
        return True
    except OSError:
        return False


def require_network(proxy: Optional[str]):
    if check_network(proxy):
        return
    sys.exit(
        "\n连不上 Gemini 接口（%s:443），评测需要调用 Embedding/生成模型。\n"
        "请任选一种方式后重跑：\n"
        "  1) 打开代理软件（Karing / Clash 等）并开启「系统代理」或 TUN 模式；\n"
        "  2) 命令行里显式指定：set HTTPS_PROXY=http://127.0.0.1:7890  （端口换成你自己的）\n"
        "  3) 直接传参：python eval_rag.py --run --proxy http://127.0.0.1:7890\n"
        % GOOGLE_HOST
    )


# ---------------------------------------------------------------- 知识库
def load_store():
    if not os.path.exists(config.VECTORSTORE_DIR) or not os.listdir(config.VECTORSTORE_DIR):
        sys.exit(f"知识库为空：{config.VECTORSTORE_DIR}。请先在应用里上传并处理文档。")
    return rag_engine.load_vectorstore()


def index_size(vectorstore) -> int:
    try:
        return vectorstore._collection.count()
    except Exception:
        data = vectorstore.get(include=[])
        return len(data.get("ids", []))


def build_retriever(vectorstore, mode: str, k: int):
    kwargs = {"k": k}
    if mode == "mmr":
        kwargs.update({"fetch_k": k * 3, "lambda_mult": 0.7})
    return vectorstore.as_retriever(search_type=mode, search_kwargs=kwargs)


# ---------------------------------------------------------------- 测试集
def load_testset(path: str) -> List[dict]:
    if not os.path.exists(path):
        sys.exit(f"找不到测试集：{path}\n可先用 --gen 生成候选测试集，或手工按脚本头部注释的格式写一个。")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list) or not data:
        sys.exit("测试集应为非空 JSON 数组")
    return data


def gen_testset(vectorstore, n_docs: int, out_path: str):
    """从知识库里抽片段，让模型为每个片段出一条"必须靠该片段才能答"的问题"""
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not config.GEMINI_API_KEY:
        sys.exit("未配置 GEMINI_API_KEY，无法自动生成测试集（可手工写 testset.json）。")

    data = vectorstore.get(include=["documents", "metadatas"])
    docs, metas = data.get("documents", []), data.get("metadatas", [])
    if not docs:
        sys.exit("知识库没有内容")

    import random

    random.seed(42)
    idxs = random.sample(range(len(docs)), min(n_docs, len(docs)))
    llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, google_api_key=config.GEMINI_API_KEY)

    prompt = """你是评测集出题人。下面是某文档的一个片段，请出一道"必须读到这个片段才能回答"的题干，
并给出答案在片段中逐字出现的一段关键证据（15-40 字，必须原样复制，不要改写）。

片段：
{chunk}

只输出 JSON：{{"question": "...", "key_phrase": "..."}}"""

    out = []
    for i, idx in enumerate(idxs, 1):
        chunk = docs[idx][:1200]
        try:
            resp = llm.invoke(prompt.format(chunk=chunk))
            text = resp.content if hasattr(resp, "content") else str(resp)
            text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
            m = re.search(r"\{.*\}", text, re.S)
            obj = json.loads(m.group(0) if m else text)
            src = (metas[idx] or {}).get("source", "?")
            page = (metas[idx] or {}).get("page", "?")
            out.append({
                "id": f"auto-{i}",
                "type": "answerable",
                "question": obj["question"].strip(),
                "key_phrase": obj["key_phrase"].strip(),
                "note": f"自动生成，请人工核对；来源 {os.path.basename(str(src))} p{page}",
            })
            print(f"[{i}/{len(idxs)}] {obj['question'][:40]}", flush=True)
        except Exception as e:
            print(f"[{i}/{len(idxs)}] 生成失败：{e}")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n已写入 {out_path}（{len(out)} 条）。请逐条人工核对 key_phrase 是否真的在原文里逐字出现，"
          f"确认无误后改名 testset.json 再跑 --run。")


# ---------------------------------------------------------------- LLM 判定
def make_judge():
    from langchain_google_genai import ChatGoogleGenerativeAI

    if not config.GEMINI_API_KEY:
        return None
    llm = ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, google_api_key=config.GEMINI_API_KEY)

    judge_prompt = """你在做 RAG 回答的事实性判定。请只依据【参考片段】判断【回答】是否有依据。

【参考片段】
{context}

【回答】
{answer}

判定规则：
- 支持：回答的每条事实信息都能在参考片段中找到依据
- 部分：部分信息有依据，但存在片段中没有的内容
- 不支持：回答包含片段中不存在的事实（编造），或片段为空却给出具体结论
只输出 JSON：{{"verdict": "支持|部分|不支持", "reason": "一句话理由"}}"""

    def judge(answer: str, contexts: List[str]) -> str:
        ctx = "\n---\n".join(c[:1500] for c in contexts[:6]) or "（无）"
        try:
            resp = llm.invoke(judge_prompt.format(context=ctx, answer=answer[:2500]))
            text = resp.content if hasattr(resp, "content") else str(resp)
            if isinstance(text, list):  # 某些版本返回 parts 列表
                text = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in text)
            return parse_verdict(text)
        except Exception:
            return "判定失败"

    return judge


def parse_verdict(text: str) -> str:
    """从模型输出里稳健地取 verdict：先抠 JSON，再退化为关键词（注意"不支持"含"支持"）"""
    m = re.search(r"\{.*\}", text or "", re.S)
    if m:
        try:
            v = json.loads(m.group(0)).get("verdict", "")
            if v:
                return v
        except Exception:
            pass
    for kw in ("判定失败", "不支持", "部分", "支持"):
        if kw in (text or ""):
            return kw
    return "判定失败"


REFUSAL_WORDS = ["没有找到", "未找到", "没有相关", "无相关", "无法从", "文档中未", "未提及",
                 "不包含", "没有提及", "无法回答", "没有提到", "文档中没有", "没有关于",
                 "未提供", "未涉及", "无此", "暂无", "没有该", "没有此类", "不涉及"]

_REFUSAL_RE = re.compile(r"(没有|未|无)[^。！？\n]{0,8}(相关|该|此|提及|涉及|提供|信息|内容|记录)")


def looks_like_refusal(answer: str) -> bool:
    a = answer or ""
    return any(w in a for w in REFUSAL_WORDS) or bool(_REFUSAL_RE.search(a))


# ---------------------------------------------------------------- 主评测
def run_eval(mode: str, ks: List[int], use_judge: bool, testset_path: str, sleep_s: float = 0.8):
    vectorstore = load_store()
    total = index_size(vectorstore)
    cases = load_testset(testset_path)
    answerable = [c for c in cases if c.get("type", "answerable") == "answerable"]
    unanswerable = [c for c in cases if c.get("type") == "unanswerable"]

    print(f"知识库片段数：{total}", flush=True)
    print(f"测试集：{len(cases)} 条（可答 {len(answerable)} / 不可答 {len(unanswerable)}）", flush=True)
    if total < 30:
        print("⚠️  知识库片段数少于 30，top-k 很容易一网打尽，Recall 会虚高，建议先入库 10 份以上文档。")

    judge = make_judge() if use_judge else None
    retr_times, gen_times = [], []
    quota_hits = 0
    per_k: Dict[int, dict] = {k: {"hits": 0, "mrr": 0.0, "n": 0} for k in ks}
    details = []

    for i, case in enumerate(cases, 1):
        q = case["question"]
        kp = case.get("key_phrase", "")
        is_unans = case.get("type") == "unanswerable"

        row = {"id": case.get("id", str(i)), "question": q, "type": case.get("type", "answerable")}

        for k in ks:
            retriever = build_retriever(vectorstore, mode, k)
            t0 = time.perf_counter()
            try:
                docs = retriever.invoke(q)
            except Exception as e:
                print(f"  检索失败：{e}")
                docs = []
            dt = (time.perf_counter() - t0) * 1000
            if k == ks[0]:
                retr_times.append(dt)

            if not is_unans and kp:
                target = norm(kp)
                rank = 0
                for pos, d in enumerate(docs, 1):
                    if target and target in norm(d.page_content):
                        rank = pos
                        break
                per_k[k]["n"] += 1
                if rank:
                    per_k[k]["hits"] += 1
                    per_k[k]["mrr"] += 1.0 / rank
                row[f"rank@{k}"] = rank
                row[f"ms@{k}"] = round(dt, 1)

        # 生成 + 判定（只用第一个 k；--no-judge 时不调用生成，省额度）
        answer, verdict = "", ""
        contexts: List[str] = []
        if judge:
            t0 = time.perf_counter()
            try:
                result = rag_engine.ask_question(q, vectorstore)
                answer = result["answer"]
                contexts = [d.page_content for d in result["source_documents"]]
                gen_times.append((time.perf_counter() - t0) * 1000)
            except Exception as e:
                err = str(e)
                row["gen_error"] = err[:200]
                if "RESOURCE_EXHAUSTED" in err or "429" in err:
                    quota_hits += 1
                    print(f"  ! 配额不足（429），本条生成失败", flush=True)
                else:
                    print(f"  ! 生成失败：{err[:120]}", flush=True)

        if judge and answer:
            verdict = judge(answer, contexts)
            row["verdict"] = verdict
        row["answer"] = answer[:600]
        if not is_unans and answer:
            row["refusal"] = looks_like_refusal(answer)

        details.append(row)
        flag = "" if is_unans else f" rank@{ks[0]}={row.get(f'rank@{ks[0]}', 0)}"
        print(f"[{i}/{len(cases)}] {q[:30]:<32}{flag}", flush=True)

        if quota_hits >= 3:
            print("\n⚠️ 连续 3 次配额不足，提前结束（已完成的部分会照常出报告）。"
                  "建议换新 Key / 等配额重置后重跑，或先用 --no-judge 只测检索。\n", flush=True)
            break
        if sleep_s:
            time.sleep(sleep_s)

    # ---------------- 汇总
    gen_failed = [r for r in details if r.get("gen_error")]
    lines = []
    lines.append(f"# RAG 评测报告\n")
    lines.append(f"- 时间：{time.strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"- 生成模型：`{config.GEMINI_MODEL}`｜Embedding：`{config.EMBEDDING_MODEL}`")
    lines.append(f"- 检索模式：`{mode}`，k = {ks}")
    lines.append(f"- 知识库片段数：{total}")
    if total < 30:
        lines.append(f"- ⚠️ 片段数仅 {total}（<30），top-k 极易覆盖全库，Recall 会虚高；建议先入库 10 份以上文档")
    lines.append(f"- 测试集：{len(cases)} 条（可答 {len(answerable)} / 不可答 {len(unanswerable)}）")
    if gen_failed:
        lines.append(f"- ⚠️ 生成失败 {len(gen_failed)} 条（配额/网络），已从耗时与忠实度统计中剔除；"
                     f"相关结论不完整，建议换 Key 或等配额重置后重跑")
    lines.append("")

    lines.append("## 检索质量\n")
    lines.append("| k | Recall@k | MRR@k |")
    lines.append("|---|---|---|")
    for k in ks:
        s = per_k[k]
        recall = s["hits"] / s["n"] if s["n"] else 0
        mrr = s["mrr"] / s["n"] if s["n"] else 0
        lines.append(f"| {k} | {recall:.1%} | {mrr:.3f} |")

    def p95(xs):
        if not xs:
            return 0
        xs = sorted(xs)
        return xs[min(len(xs) - 1, int(round(0.95 * (len(xs) - 1))))]

    lines.append("\n## 耗时（毫秒）\n")
    lines.append("| 阶段 | 均值 | 中位数 | P95 | 样本数 |")
    lines.append("|---|---|---|---|---|")
    if retr_times:
        lines.append(f"| 检索（k={ks[0]}） | {statistics.mean(retr_times):.0f} | "
                     f"{statistics.median(retr_times):.0f} | {p95(retr_times):.0f} | {len(retr_times)} |")
    if gen_times:
        lines.append(f"| 端到端问答 | {statistics.mean(gen_times):.0f} | "
                     f"{statistics.median(gen_times):.0f} | {p95(gen_times):.0f} | {len(gen_times)} |")

    if unanswerable and judge:
        answered = [r for r in details if r["type"] == "unanswerable" and not r.get("gen_error")]
        refused = sum(1 for r in answered if looks_like_refusal(r.get("answer", "")))
        ok_rate = f"{refused / len(answered):.0%}" if answered else "N/A"
        lines.append(f"\n## 拒答\n\n应当拒答 {len(unanswerable)} 条，实际完成生成 {len(answered)} 条，"
                     f"其中明确表示「文档中没有」 {refused} 条，拒答正确率 {ok_rate}")

    if judge:
        verds = [r.get("verdict", "") for r in details
                 if r["type"] == "answerable" and r.get("verdict") in ("支持", "部分", "不支持")]
        unsupported = sum(1 for v in verds if v == "不支持")
        partial = sum(1 for v in verds if v == "部分")
        lines.append(f"\n## 回答忠实度（LLM 判定，有效样本 n={len(verds)}）\n")
        lines.append(f"- 支持：{verds.count('支持')}｜部分：{partial}｜不支持（疑似编造）：{unsupported}")
        if verds:
            lines.append(f"- 无依据回答率：{unsupported / len(verds):.0%}")
        if gen_failed:
            lines.append(f"- 另有 {len(gen_failed)} 条因生成失败未判定")

    report = "\n".join(lines) + "\n"
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    print("\n" + report)
    print(f"报告：{REPORT_PATH}\n明细：{RESULT_PATH}")


def main():
    ap = argparse.ArgumentParser(description="RAG 检索与问答评测")
    ap.add_argument("--run", action="store_true", help="执行评测")
    ap.add_argument("--chunks", action="store_true", help="只打印知识库片段数")
    ap.add_argument("--mode", default="mmr", choices=["mmr", "similarity"], help="检索模式")
    ap.add_argument("--k", action="append", type=int, help="可多次传入，例如 --k 3 --k 6 --k 9")
    ap.add_argument("--no-judge", action="store_true", help="不做 LLM 判定（只测检索与耗时）")
    ap.add_argument("--testset", default=TESTSET_PATH, help="测试集路径")
    ap.add_argument("--gen", type=int, metavar="N", help="从知识库自动生成 N 条候选测试集")
    ap.add_argument("--proxy", default=None, help="HTTP 代理，例如 http://127.0.0.1:7890")
    ap.add_argument("--sleep", type=float, default=0.8, help="每条用例之间的间隔秒数（避免触发限流，默认 0.8）")
    args = ap.parse_args()

    proxy = setup_proxy(args.proxy)

    if args.chunks:
        vs = load_store()
        print(f"知识库片段数：{index_size(vs)}")
        return
    if args.gen:
        require_network(proxy)
        gen_testset(load_store(), args.gen, os.path.join(config.BASE_DIR, "testset_draft.json"))
        return
    if not args.run:
        ap.print_help()
        return

    require_network(proxy)
    run_eval(args.mode, args.k or [6], not args.no_judge, args.testset, args.sleep)


if __name__ == "__main__":
    main()
