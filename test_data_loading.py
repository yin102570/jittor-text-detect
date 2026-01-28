#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试数据加载逻辑（不依赖Jittor，用于验证样本数量问题）
"""

import json
import sys
import os

# 复制 run.py 中的数据加载函数
def load_builtin_data_with_labels(max_raw_data=200, min_samples=10):
    """
    加载带标签的内置数据（简化版，不依赖argparse）
    """
    # 🟢 明确的人类文本（维基百科片段）
    human_texts = [
        "The Great Barrier Reef is the world's largest coral reef system, located in the Coral Sea off the coast of Australia.",
        "Python is a high-level, general-purpose programming language designed for readability and ease of use.",
        "Photosynthesis is the process by which plants use sunlight to convert carbon dioxide and water into glucose and oxygen.",
        "The Roman Empire was one of the largest empires in history, spanning across Europe, Africa, and Asia.",
        "Albert Einstein developed the theory of relativity, which revolutionized our understanding of space and time.",
        "The Internet is a global network of interconnected computer networks that use the Internet protocol suite to communicate.",
        "DNA, or deoxyribonucleic acid, is the molecule that carries genetic information for all living organisms.",
        "Shakespeare wrote 39 plays, including tragedies like Hamlet, comedies like A Midsummer Night's Dream, and histories like Henry V.",
        "The Industrial Revolution began in Great Britain in the late 18th century, transforming agrarian societies into industrial ones.",
        "Mount Everest is the highest mountain on Earth, with a peak at 8,848 meters above sea level.",
        "Water is a polar molecule composed of two hydrogen atoms and one oxygen atom, essential for all known forms of life.",
        "The French Revolution began in 1789, overthrowing the monarchy and establishing a republic in France.",
        "Quantum mechanics is a branch of physics that describes the behavior of matter and energy at the atomic and subatomic level.",
        "Amazon River is the largest river by discharge volume of water in the world, located in South America.",
        "Vincent van Gogh was a Dutch post-impressionist painter known for works like Starry Night and Sunflowers.",
        "The Moon is Earth's only natural satellite, orbiting at an average distance of 384,400 kilometers.",
        "Coffee is a brewed drink prepared from roasted coffee beans, the seeds of berries from certain Coffea species.",
        "The Renaissance was a period of European cultural, artistic, political, and economic rebirth following the Middle Ages.",
        "Electricity is the set of physical phenomena associated with the presence and motion of electric charge.",
        "Pandas are a bear species native to China, known for their distinctive black-and-white coat and diet of bamboo.",
        "The United Nations was founded in 1945 to promote international cooperation and maintain peace and security.",
        "Classical music refers to the art music of the Western world, including composers like Beethoven, Mozart, and Bach.",
        "Volcanoes are ruptures in the crust of a planetary-mass object, allowing hot lava, volcanic ash, and gases to escape.",
        "The human brain is the central organ of the human nervous system, responsible for thought, memory, and emotion.",
        "Bitcoin is a decentralized digital currency, without a central bank or single administrator, that can be sent from user to user.",
        "The Sahara Desert is the largest hot desert in the world, covering most of North Africa.",
        "Marie Curie was a Polish-French physicist and chemist who conducted pioneering research on radioactivity, winning two Nobel Prizes.",
        "Rice is the seed of the grass species Oryza sativa or Oryza glaberrima, a staple food for more than half the world's population.",
        "The Internet of Things (IoT) refers to physical objects embedded with sensors, software, and connectivity to exchange data.",
        "Sharks are a group of elasmobranch fish characterized by a cartilaginous skeleton, five to seven gill slits on the sides of the head.",
        "The Louvre Museum in Paris is the world's largest art museum, housing works like the Mona Lisa and Venus de Milo.",
        "Climate change refers to long-term shifts in temperatures and weather patterns, largely caused by human activities.",
        "Basketball is a team sport played on a rectangular court, where two teams of five players aim to shoot a ball through a hoop.",
        "The human heart is a muscular organ that pumps blood through the circulatory system, supplying oxygen and nutrients to tissues.",
        "Tokyo is the capital and most populous city of Japan, known for its skyscrapers, shopping districts, and cultural landmarks.",
        "Plastics are a wide range of synthetic or semi-synthetic materials that use polymers as a main ingredient.",
        "The Olympics are a series of international multi-sport events held every four years, featuring summer and winter games.",
        "Gravity is a natural phenomenon by which all things with mass or energy are attracted to one another.",
        "Chocolate is a food made from roasted and ground cacao seeds, originating from Mesoamerica.",
        "The telephone was invented by Alexander Graham Bell, revolutionizing long-distance communication.",
        "Forests cover approximately 31% of the world's land area, providing habitat for millions of species.",
        "Artificial intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.",
        "The Nile River is a major north-flowing river in northeastern Africa, the longest river in the world.",
        "Jazz is a music genre that originated in the late 19th and early 20th centuries in the Southern United States.",
        "Diamonds are allotrope of carbon where the carbon atoms are arranged in a crystal structure called diamond cubic.",
        "The human eye is an organ that reacts to light and allows vision, detecting color, brightness, and movement.",
        "Space exploration is the use of astronomy and space technology to explore outer space, including missions to the Moon and Mars.",
        "Tea is an aromatic beverage prepared by pouring hot or boiling water over cured or fresh leaves of Camellia sinensis.",
        "Democracy is a form of government in which power is held by the people, either directly or through elected representatives.",
        "Volleyball is a team sport in which two teams of six players are separated by a net, using their hands to hit a ball over the net."
    ]

    # 🔵 明确的AI生成文本（短故事）
    ai_texts = [
        "Lila forgot her umbrella on the bus, but a stranger shared theirs and they became friends.",
        "Jake practiced the guitar for months and finally played at the local café to a cheering crowd.",
        "Mia found a lost dog in the park, tracked down its owner, and was given a homemade pie as a thank you.",
        "Tom planted a seed in his backyard and watched it grow into a cherry tree over three years.",
        "Zoe missed her train but met an old friend at the station, making the delay worthwhile.",
        "Ben volunteered at the animal shelter and adopted a shy kitten that quickly became his best friend.",
        "Luna baked cookies for her neighbor who was sick, and they ended up sharing stories all afternoon.",
        "Max found a vintage book at a garage sale, discovered it was signed by the author, and donated it to the library.",
        "Sophie taught her little brother to ride a bike, and he surprised her by riding alone the next day.",
        "Eli saved up for a new camera and took the perfect photo of a sunset over the lake.",
        "Clara lost her favorite necklace but found it while cleaning her room, hidden under a pile of books.",
        "Jesse helped an elderly neighbor carry groceries, and they invited him for dinner every week after that.",
        "Maya joined a painting class and sold her first artwork at a local gallery.",
        "Leo forgot his lunch at home, but his classmates shared their food, making him feel welcome.",
        "Nora wrote a letter to her pen pal in another country and received a reply with photos of their hometown.",
        "Owen fixed his grandma's old radio, and she played her favorite songs while they baked cookies.",
        "Piper found a four-leaf clover in the park and gave it to her mom who was having a bad day.",
        "Quinn organized a book drive for the school library and collected over 100 books.",
        "Riley learned to cook pasta from their dad and made dinner for the whole family that night.",
        "Sam found a wallet on the street, returned it to its owner, and refused a reward, saying it was the right thing to do.",
        "Tina tried out for the school play and got the lead role, even though she was nervous to audition.",
        "Umar taught his dog to fetch a ball, and they spent every afternoon playing in the park.",
        "Violet grew tomatoes in her window box and shared them with her entire apartment building.",
        "Will found an old photo album in the attic and asked his grandparents to tell stories about the pictures.",
        "Xena joined the school debate team and won her first competition, surprising even herself.",
        "Yusuf donated his old clothes to charity and met a kid who loved the jacket he gave away.",
        "Zara wrote a poem for her teacher, who read it aloud to the class and praised her creativity.",
        "Adam built a birdhouse with his dad and watched a family of sparrows move in within a week.",
        "Bella tried sushi for the first time and loved it, then took her parents to the same restaurant.",
        "Charlie missed the school bus but ran all the way and arrived just in time for his math test.",
        "Daisy started a journal and wrote in it every night, finding comfort in putting her thoughts on paper.",
        "Ethan helped his sister with her homework, and she aced her science exam the next day.",
        "Fiona found a seashell on the beach that reminded her of her summer vacation with her grandma.",
        "George learned to play chess from his grandpa and beat him for the first time on his birthday.",
        "Hannah left her phone at the mall, but a store clerk kept it safe and called her to pick it up.",
        "Ian planted flowers in the community garden and saw bees and butterflies visit every day.",
        "Julia saved a butterfly with a damaged wing, and it flew away after a week of care.",
        "Kai joined a soccer team and scored the winning goal in his first game.",
        "Liam found a box of old comics in his basement and sold them to a collector for enough to buy a new bike.",
        "Molly wrote a short story and entered it in a contest, winning a gift card to a bookstore.",
        "Noah helped his mom plant a vegetable garden, and they ate fresh carrots all summer long.",
        "Olivia forgot her lines in the school play but ad-libbed and got a laugh from the audience.",
        "Paul found a meteorite fragment while hiking and showed it to his science teacher, who was impressed.",
        "Quincy learned to juggle and performed at his little sister's birthday party, making her laugh.",
        "Rachel donated blood for the first time and found out her blood type helped a sick child.",
        "Simon fixed his bike's flat tire by himself and rode it to the park to meet friends.",
        "Tara made a scrapbook of her summer vacation and gave it to her best friend as a gift.",
        "Uma found a rare flower in the woods and took a photo for her nature project at school.",
        "Victor practiced piano every day and played a song at his grandma's 80th birthday party.",
        "Wendy volunteered to read to kids at the library and discovered she loved storytelling."
    ]

    # 合并所有文本
    all_human_texts = human_texts
    all_ai_texts = ai_texts

    # 根据参数决定使用多少样本
    n_samples = min(max_raw_data // 2, len(all_human_texts), len(all_ai_texts))

    # 确保至少min_samples个样本
    n_samples = max(n_samples, min_samples)

    # 选取前n_samples个样本
    selected_human = all_human_texts[:n_samples]
    selected_ai = all_ai_texts[:n_samples]

    print(f"[OK] 加载带标签数据：人类文本 {len(selected_human)} 条，AI文本 {len(selected_ai)} 条")
    print(f"[INFO] 总样本数：{len(selected_human) + len(selected_ai)} 条")

    # 🎯 关键修复：DetectGPT需要original=人类文本，samples=AI文本
    return {
        "original": selected_human,  # 人类文本（标签0）
        "samples": selected_ai,  # AI文本（标签1）
        "labels": [0] * len(selected_human) + [1] * len(selected_ai),
        "human": selected_human,
        "ai": selected_ai
    }


def check_data_validity(data, min_samples=20):
    """检查数据有效性"""
    print(f"\n{'='*60}")
    print("数据有效性检查")
    print(f"{'='*60}")

    if isinstance(data, dict):
        # 获取原始文本和样本文本
        original_count = len(data.get("original", []))
        samples_count = len(data.get("samples", []))

        # 总样本数是两者之和
        total_samples = original_count + samples_count

        print(f"[INFO] 数据统计:")
        print(f"   - original（人类文本）: {original_count} 条")
        print(f"   - samples（AI文本）: {samples_count} 条")
        print(f"   - 总计: {total_samples} 条")

        if total_samples == 0:
            print("[ERROR] 错误: 数据为空！")
            return False

        # 检查样本数量是否满足要求
        if total_samples < min_samples:
            print(f"[ERROR] 错误: 样本数量不足 (需要≥{min_samples}，当前{total_samples})")
            return False

        # 检查标签数量是否匹配
        labels_count = len(data.get("labels", []))
        if labels_count != total_samples:
            print(f"[WARN] 警告: 标签数量不匹配 (文本{total_samples}条，标签{labels_count}条)")
        else:
            print(f"[OK] 标签数量匹配: {labels_count} 条")

        # 检查标签分布
        if labels_count > 0:
            labels = data.get("labels", [])
            label_0_count = labels.count(0)
            label_1_count = labels.count(1)
            print(f"[OK] 标签分布: 标签0（人类）{label_0_count}条, 标签1（AI）{label_1_count}条")

        # 检查文本长度
        print(f"\n[INFO] 文本长度分析:")
        if original_count > 0:
            human_lengths = [len(t.split()) for t in data["original"]]
            print(f"   人类文本: 平均 {sum(human_lengths)/len(human_lengths):.1f} 词, 最短 {min(human_lengths)} 词, 最长 {max(human_lengths)} 词")

        if samples_count > 0:
            ai_lengths = [len(t.split()) for t in data["samples"]]
            print(f"   AI文本: 平均 {sum(ai_lengths)/len(ai_lengths):.1f} 词, 最短 {min(ai_lengths)} 词, 最长 {max(ai_lengths)} 词")

        print(f"\n[OK] 数据格式有效")
        return True
    else:
        print("[ERROR] 错误: 数据不是字典格式")
        return False


def test_data_cleaning(data):
    """模拟 detectGPT.py 中的文本清理逻辑"""
    print(f"\n{'='*60}")
    print("模拟文本清理（使用 >10 字符过滤）")
    print(f"{'='*60}")

    original_texts = data.get("original", [])
    sampled_texts = data.get("samples", [])

    cleaned_original = []
    cleaned_samples = []

    for i, (o, s) in enumerate(zip(original_texts, sampled_texts)):
        valid_o = isinstance(o, str) and o.strip() and len(o.strip()) > 10
        valid_s = isinstance(s, str) and s.strip() and len(s.strip()) > 10

        if valid_o and valid_s:
            cleaned_original.append(o.strip())
            cleaned_samples.append(s.strip())
        else:
            print(f"[WARN] 跳过无效样本 #{i + 1}: 原始={valid_o}, 生成={valid_s}")

    print(f"\n清理后统计:")
    print(f"   人类文本: {len(original_texts)} -> {len(cleaned_original)}")
    print(f"   AI文本: {len(sampled_texts)} -> {len(cleaned_samples)}")
    print(f"   总计: {len(cleaned_original) + len(cleaned_samples)}")

    return cleaned_original, cleaned_samples


if __name__ == "__main__":
    print("="*60)
    print("Jittor DetectGPT 数据加载测试")
    print("="*60)

    # 测试不同参数组合
    test_cases = [
        (200, 10),   # 最大样本数，最小要求10
        (100, 10),   # 中等样本数
        (40, 10),    # 较小样本数
        (20, 10),    # 最小样本数
    ]

    for max_raw_data, min_samples in test_cases:
        print(f"\n\n{'#'*60}")
        print(f"测试配置: max_raw_data={max_raw_data}, min_samples={min_samples}")
        print(f"{'#'*60}")

        data = load_builtin_data_with_labels(max_raw_data, min_samples)

        if check_data_validity(data, min_samples=min_samples):
            cleaned_human, cleaned_ai = test_data_cleaning(data)

            if len(cleaned_human) >= 2 and len(cleaned_ai) >= 2:
                print(f"\n[OK] 测试通过: 样本数量充足")
            else:
                print(f"\n[ERROR] 测试失败: 样本数量不足（至少需要2个）")
        else:
            print(f"\n[ERROR] 测试失败: 数据验证未通过")

    print(f"\n\n{'='*60}")
    print("所有测试完成")
    print(f"{'='*60}")
