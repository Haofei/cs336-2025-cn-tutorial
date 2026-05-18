#!/usr/bin/env python3
from __future__ import annotations

import html
import re
import subprocess
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile


ROOT = Path(__file__).resolve().parent
FRONTMATTER = ROOT / "language-models-from-scratch-frontmatter.md"
MAIN = ROOT / "language-models-from-scratch-main.md"
OUTPUT = ROOT / "language-models-from-scratch.epub"
CSS = ROOT / "epub-assets" / "epub.css"
COVER = ROOT / "epub-assets" / "cover.svg"

TITLE = "从零理解语言模型"
AUTHOR = "江理子"
LANG = "zh-CN"


SECTION_INSERTIONS = {
    "# 序章：为什么要从零理解语言模型": """# 第一部分：从文字到模型内部

这一部分先把资料助手压到最小形态：用户给它一段文字，它要把文字变成 token，把 token 变成张量，再通过 Transformer 建立上下文联系。读完这一部分，读者不需要已经会训练模型，但应该能看懂“文本进入模型”这条路径。

这里的重点是建立共同语言。后面所有训练、推理、评估和对齐问题，都会反复回到 token、shape、attention、position 和 loss 这些底层对象。""",
    "# 第 5 章：从文本数据到训练样本": """# 第二部分：把模型训练起来

第一部分解释了模型内部怎么处理 token。现在问题变成：这些 token 从哪里来，怎样变成训练样本，一次训练 step 到底更新了什么，以及为什么训练成本会迅速变大。

对资料助手来说，这一部分对应“从想法到实验系统”。如果没有可检查的数据、训练循环、日志、checkpoint 和资源账本，后面的能力改进都无法复盘。""",
    "# 第 9 章：Triton 与高性能算子的直觉": """# 第三部分：让系统跑得动

模型能训练起来，还不等于能以合理成本服务用户。这一部分把视角从单次训练 step 放大到算子、多卡、容量、预算和推理服务。

资料助手在这里第一次变成真实系统：它要读长资料，要服务并发用户，要控制延迟和显存。nanoGPT 仍然是最小参照，但从这一部分开始，很多问题已经超出最小代码库，必须用系统工程补上。""",
    "# 第 14 章：评估：不要被一个分数骗了": """# 第四部分：让回答可信

系统跑得动以后，更难的问题是：回答到底能不能信。资料助手不是生成一段流畅文字就算成功，它必须知道资料来源、承认边界、引用证据、接受评估，并在必要时通过 SFT、偏好学习和可验证反馈改变行为。

这一部分也是从 nanoGPT 到真实产品的桥。nanoGPT 能说明预训练闭环，却不包含完整的数据治理、评估集、SFT、RLHF 或 RLVR 管线。读这些章节时，要把 nanoGPT 当成底座，而不是把它当成全部系统。""",
    "# 第 21 章：一个 tiny language model 项目": """# 第五部分：把理解变成项目

前面四部分把系统拆开看，这一部分把它重新合上。读者需要用一个小项目检查自己是否真的理解了链路，再决定下一步是做应用、做训练系统，还是继续读研究。

资料助手在这里不再只是案例，而是一张路线图：任何新技术都可以放回数据、模型、训练、推理、评估、对齐和产品边界中判断。""",
    "# 后记：从书到实践": "# 后记与实践",
    "# 附录 A：核心术语表": "# 附录",
}

HEADING_RENAMES = {}

PART_TITLES = {
    "第一部分：从文字到模型内部",
    "第二部分：把模型训练起来",
    "第三部分：让系统跑得动",
    "第四部分：让回答可信",
    "第五部分：把理解变成项目",
    "后记与实践",
    "附录",
}


def build_markdown(build_md: Path) -> None:
    front_lines = FRONTMATTER.read_text(encoding="utf-8").splitlines()
    if front_lines and front_lines[0].startswith("# "):
        front_lines[0] = "# 序言"

    out_lines = ["\n".join(front_lines).rstrip(), ""]
    for line in MAIN.read_text(encoding="utf-8").splitlines():
        if line in SECTION_INSERTIONS:
            out_lines.extend(["", SECTION_INSERTIONS[line], ""])

        if line in HEADING_RENAMES:
            out_lines.append(HEADING_RENAMES[line])
        elif line.startswith("# "):
            out_lines.append("#" + line)
        else:
            out_lines.append(line)

    build_md.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")


def run_pandoc(build_md: Path) -> None:
    subprocess.run(
        [
            "pandoc",
            str(build_md),
            "-o",
            str(OUTPUT),
            "--metadata",
            f"title={TITLE}",
            "--metadata",
            f"author={AUTHOR}",
            "--metadata",
            f"lang={LANG}",
            "--toc",
            "--toc-depth=2",
            "--split-level=2",
            f"--css={CSS}",
            f"--epub-cover-image={COVER}",
        ],
        check=True,
        cwd=ROOT,
    )


def strip_tags(value: str) -> str:
    value = re.sub(r"<[^>]+>", "", value)
    return html.unescape(value).strip()


def section_type_for(title: str, level: str) -> str:
    if title == "序言":
        return "preface"
    if title in PART_TITLES and level == "1":
        return "part"
    if title.startswith("附录"):
        return "appendix"
    if title.startswith("序章") or re.match(r"第\s*\d+\s*章", title):
        return "chapter"
    if level == "2":
        return "chapter"
    return "part"


def postprocess_xhtml(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    match = re.search(r"<h([12])[^>]*>(.*?)</h\1>", text, flags=re.S)
    if not match:
        return

    level = match.group(1)
    title = strip_tags(match.group(2))
    escaped_title = html.escape(title, quote=False)
    text = re.sub(r"<title>.*?</title>", f"<title>{escaped_title}</title>", text, count=1, flags=re.S)

    epub_type = section_type_for(title, level)
    text = re.sub(
        r"(<section\b)(?![^>]*\bepub:type=)([^>]*>)",
        rf'\1 epub:type="{epub_type}"\2',
        text,
        count=1,
    )
    path.write_text(text, encoding="utf-8")


def repack_epub(work_dir: Path, output: Path) -> None:
    rebuilt = output.with_suffix(".rebuilt.epub")
    with ZipFile(rebuilt, "w") as zf:
        mimetype = work_dir / "mimetype"
        if mimetype.exists():
            zf.write(mimetype, "mimetype", compress_type=ZIP_STORED)

        for path in sorted(work_dir.rglob("*")):
            if path.is_dir() or path.name == "mimetype":
                continue
            zf.write(path, path.relative_to(work_dir).as_posix(), compress_type=ZIP_DEFLATED)

    rebuilt.replace(output)


def postprocess_epub() -> None:
    with tempfile.TemporaryDirectory(prefix="lmfs-epub-") as tmp:
        work_dir = Path(tmp)
        with ZipFile(OUTPUT, "r") as zf:
            zf.extractall(work_dir)

        for path in sorted((work_dir / "EPUB" / "text").glob("ch*.xhtml")):
            postprocess_xhtml(path)

        repack_epub(work_dir, OUTPUT)


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="lmfs-md-") as tmp:
        build_md = Path(tmp) / "book.md"
        build_markdown(build_md)
        run_pandoc(build_md)
    postprocess_epub()


if __name__ == "__main__":
    main()
