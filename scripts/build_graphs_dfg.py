import json
import re
from pathlib import Path
from tqdm import tqdm

# ============================
# 路径：使用合约级 CFG !!!
# ============================
AST_DIR = Path("data/graphs_ast_llm")
CFG_DIR = Path("data/graphs_cfg_contract")
OUT_DIR = Path("data/graphs_dfg")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 链名称（你已有的六个链）
CHAINS = ["BSC", "Ethereum", "Polygon", "Avalanche", "Fantom", "Arbitrum"]

# ============================
# 简单变量名正则
# ============================
IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")

SOLIDITY_KEYWORDS = {
    "if", "else", "for", "while", "return", "emit",
    "require", "revert", "assert",
    "mapping", "struct", "event", "function",
    "pragma", "contract", "interface", "library",
    "memory", "storage", "calldata", "public",
    "private", "internal", "external", "view",
    "pure", "payable", "virtual", "override",
    "returns", "modifier", "using", "new", "delete",
}


def normalize_cid(cid: str) -> str:
    """
    统一 AST/CFG 的 contract id：

    AST 例子（你现在的情况）:
      data/raw/BSC/145fe1...b488f_CAKEBACK.sol

    CFG 例子:
      145fe1...b488f_CAKEBACK
      6e211fA...D1B_TimedToken   (地址大小写混用)

    归一化输出:
      <addr_lower>_<contractName>    （不带路径，不带 .sol）
    """
    if cid is None:
        return ""
    s = str(cid).strip()
    if not s:
        return ""

    # basename: 去掉目录
    s = Path(s).name

    # 去掉 .sol 后缀
    if s.lower().endswith(".sol"):
        s = s[:-4]

    # 地址部分统一小写（只处理 '_' 前面那段）
    if "_" in s:
        addr, rest = s.split("_", 1)
        s = addr.lower() + "_" + rest
    else:
        s = s.lower()

    return s


def extract_vars_from_expr(expr: str):
    """从 CFG 节点的 expression 中抽变量"""
    if not expr or not isinstance(expr, str):
        return []

    tokens = IDENT_RE.findall(expr)
    results = []
    for t in tokens:
        if t.lower() in SOLIDITY_KEYWORDS:
            continue
        results.append(t)
    return results


# --------------------------------------------------------
# 加载“合约级 CFG”，非常关键
# --------------------------------------------------------
def load_cfg_by_id(chain: str):
    """读取 data/graphs_cfg_contract/<chain>.jsonl，返回 normalized_id -> cfg_nodes 映射"""
    cfg_path = CFG_DIR / f"{chain}.jsonl"
    if not cfg_path.exists():
        print(f"[WARN] No contract-level CFG for {chain}: {cfg_path}")
        return {}

    id2nodes = {}

    with cfg_path.open("r", encoding="utf8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue

            gid_raw = item.get("id")
            if gid_raw is None:
                continue

            gid = normalize_cid(gid_raw)
            if not gid:
                continue

            # ⬇⬇⬇ 重点：合约级 CFG 节点字段
            nodes = item.get("cfg_nodes", [])
            if isinstance(nodes, list):
                id2nodes[gid] = nodes

    print(f"[INFO] Loaded CFG for {chain}, contracts = {len(id2nodes)}")
    return id2nodes


# --------------------------------------------------------
# 以 CFG 节点构建简单 DFG（变量流）
# --------------------------------------------------------
def build_dfg_from_cfg_nodes(cfg_nodes):
    var2nodes = {}

    for n in cfg_nodes:
        nid = n.get("id")
        expr = n.get("expression")
        vars_ = extract_vars_from_expr(expr)

        for v in vars_:
            var2nodes.setdefault(v, []).append(nid)

    # 构建 DFG 边
    dfg_edges = []
    for var, lst in var2nodes.items():
        # 去重并保持顺序
        seen = set()
        new_list = []
        for x in lst:
            if x not in seen:
                seen.add(x)
                new_list.append(x)

        # 连链式
        for i in range(len(new_list) - 1):
            dfg_edges.append({
                "src": new_list[i],
                "dst": new_list[i + 1],
                "var": var
            })

    # DFG 节点（直接复用 CFG 节点）
    dfg_nodes = []
    for n in cfg_nodes:
        dfg_nodes.append({
            "id": n.get("id"),
            "type": n.get("type"),
            "expression": n.get("expression"),
            "contract": n.get("contract"),
            "function": n.get("function"),
        })

    return dfg_nodes, dfg_edges


# --------------------------------------------------------
# 主流程
# --------------------------------------------------------
def main():
    print("[INFO] Building DFG from contract-level CFG...")

    for chain in CHAINS:
        ast_path = AST_DIR / f"{chain}.jsonl"
        if not ast_path.exists():
            print(f"[WARN] No AST for {chain}, skip. ({ast_path})")
            continue

        # 加载合约级 CFG（按 normalized id 建索引）
        cfg_index = load_cfg_by_id(chain)
        if not cfg_index:
            print(f"[WARN] No CFG loaded for {chain}, skip.")
            continue

        out_path = OUT_DIR / f"{chain}.jsonl"

        total_ast = 0
        matched = 0
        skipped = 0

        with ast_path.open("r", encoding="utf8") as fin, \
             out_path.open("w", encoding="utf8") as fout:

            for line in tqdm(fin, desc=f"DFG {chain}"):
                line = line.strip()
                if not line:
                    continue

                try:
                    ast_item = json.loads(line)
                except Exception:
                    continue

                total_ast += 1

                gid_raw = ast_item.get("id")
                gid = normalize_cid(gid_raw)

                if not gid or gid not in cfg_index:
                    skipped += 1
                    continue

                cfg_nodes = cfg_index[gid]
                dfg_nodes, dfg_edges = build_dfg_from_cfg_nodes(cfg_nodes)

                fout.write(json.dumps({
                    "id": gid,          # ✅ 写回归一化后的 id，后续永远一致
                    "chain": chain,
                    "dfg_nodes": dfg_nodes,
                    "dfg_edges": dfg_edges
                }) + "\n")

                matched += 1

        print(
            f"[OK] DFG built for {chain}, total = {matched} | "
            f"AST_seen={total_ast} matched={matched} skipped={skipped} | "
            f"out={out_path}"
        )


if __name__ == "__main__":
    main()