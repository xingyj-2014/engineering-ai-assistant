file_path = r"C:\Users\Administrator\Desktop\我的工程AI知识库\guifan.txt"

# 1. 先把文件读进来（就像你刚才 step1 做的那样）
with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# 2. 按段落切碎
paragraphs = content.split("\n")

# 3. 让用户输入一个要查的关键词
keyword = input("请输入你要查的关键词（比如：安全）：")

# 4. 准备一个空列表，用来装找到的段落
results = []

# 5. 循环检查每一段
for p in paragraphs:
    if keyword in p:  # 如果关键词在这一段里
        results.append(p)  # 就把这一段装进结果列表

# 6. 看看找没找到
if len(results) == 0:
    print(f"\n抱歉，规范中没有找到包含【{keyword}】的内容。")
else:
    print(f"\n=== 找到了 {len(results)} 段包含【{keyword}】的内容 ===")
    for r in results:
        print("-", r)