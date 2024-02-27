import json

def main():
    new_file = 'data/dataset_info.json'

    # 读取 JSON 文件
    with open(new_file, 'r') as f:
        data = json.load(f)

    # 定义要添加的文件名
    file_names = ["Chinese_erotic.jsonl", "Chinese_NovelWriting.jsonl", "merge_cut3k_121k_no_source.jsonl","Identity_data_for_0.3.jsonl",\
                  "novel_50_to_profile_and_convs_part0.jsonl","novel_50_to_profile_and_convs_part1.jsonl","translated_and_split_PIPPA.jsonl"]

    # 添加新信息
    for i, file_name in enumerate(file_names):
        data[f'haruhi_zero_{i}'] = {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {
                "messages": "conversations",
                "system": "system",
            },
            "tags": {
                "role_tag": "from",
                "content_tag": "value",
                "user_tag": "human",
                "assistant_tag": "gpt",
            }
        }

    # 将修改后的数据写回文件
    with open(new_file, 'w') as f:
        json.dump(data, f, indent=2)

    print('Data copied and new info appended to', new_file)

if __name__ == "__main__":
    main()
