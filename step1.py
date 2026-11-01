file_path = r"C:\Users\Administrator\Desktop\我的工程AI知识库\guifan.txt"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

print("=== 成功读取文件 ===")
print("总字数：", len(content))

paragraphs = content.split("\n")
print("一共切出了", len(paragraphs), "段")