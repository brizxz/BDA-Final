import os
import subprocess

try:
    # 檢查是否安裝了pandoc
    subprocess.run(["pandoc", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print("Pandoc已安裝，開始轉換...")
except:
    print("請先安裝Pandoc: https://pandoc.org/installing.html")
    exit(1)

# 轉換Markdown到PDF
try:
    subprocess.run([
        "pandoc", 
        "r119020XX_report.md", 
        "-o", "r119020XX_report.pdf",
        "--pdf-engine=xelatex",
        "-V", "geometry:margin=1in",
        "-V", "mainfont:DejaVu Sans",
        "--toc"
    ], check=True)
    print("轉換成功！已生成 r119020XX_report.pdf")
except Exception as e:
    print(f"轉換失敗: {e}")
    print("請確保已安裝LaTeX或嘗試手動轉換。") 