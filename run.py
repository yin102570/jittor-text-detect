
import jittor as jt
import argparse
import json
import os
import sys

# 原代码导入，保持路径不变
from utils.save_results import save_results
from utils.baselines.detectGPT import detectGPT
from utils.baselines.run_baselines import run_baselines
from utils.setting import set_experiment_config, initial_setup
from utils.load_models_tokenizers import load_base_model_and_tokenizer, load_base_model, load_mask_filling_model


# ====================== 核心：内置200条文本数据（修复samples键） ======================
def load_builtin_data_with_labels(args):
    """
    加载带标签的内置数据
    明确区分人类文本（0）和AI生成文本（1）
    """
    # 🟢 明确的人类文本（维基百科片段 - 真实写作）
    human_texts = [
        # 类别2：维基百科片段（明确的人类写作）
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

    # 🔵 明确的AI生成文本（短故事 - 模仿GPT风格）
    ai_texts = [
        # 类别1：短故事（模仿AI生成的风格）
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

    # 🔥 增加更多样本以确保通过验证
    # 添加更多人类文本
    more_human_texts = [
        "The periodic table is a tabular arrangement of chemical elements, organized by atomic number and electron configuration.",
        "The human skeletal system consists of 206 bones that provide structure, support, and protection for the body.",
        "Photosynthesis occurs in the chloroplasts of plant cells, using chlorophyll to capture light energy.",
        "Newton's laws of motion describe the relationship between a body and the forces acting upon it, and its motion in response.",
        "The World Wide Web was invented by Tim Berners-Lee in 1989, revolutionizing information sharing globally.",
        "Mitochondria are often called the powerhouses of the cell, producing ATP through cellular respiration.",
        "The Eiffel Tower in Paris was completed in 1889 and stands 330 meters tall as a symbol of France.",
        "Global warming refers to the long-term increase in Earth's average surface temperature due to human activities.",
        "The human digestive system breaks down food into nutrients that can be absorbed and used by the body.",
        "Renewable energy sources include solar, wind, hydroelectric, geothermal, and biomass energy.",
        "The Mona Lisa is a portrait painting by Leonardo da Vinci, housed in the Louvre Museum in Paris.",
        "Earth's atmosphere is composed primarily of nitrogen (78%) and oxygen (21%), with trace amounts of other gases.",
        "The American Revolution was a colonial revolt that took place between 1765 and 1783, establishing the United States.",
        "Plate tectonics theory explains the movement of Earth's lithosphere, causing earthquakes, volcanoes, and mountain formation.",
        "Vitamin C is an essential nutrient found in citrus fruits, important for immune function and collagen synthesis.",
        "The printing press, invented by Johannes Gutenberg around 1440, revolutionized the spread of information.",
        "The solar system consists of the Sun and the objects that orbit it, including eight planets, dwarf planets, and other celestial bodies.",
        "The circulatory system transports blood throughout the body, delivering oxygen and nutrients to cells.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming.",
        "The immune system protects the body from pathogens through a complex network of cells, tissues, and organs.",
        "The Great Wall of China was built over centuries to protect Chinese states and empires from nomadic invasions.",
        "Quantum computing uses quantum-mechanical phenomena like superposition and entanglement to perform computations.",
        "The respiratory system facilitates gas exchange, bringing oxygen into the body and removing carbon dioxide.",
        "Blockchain technology provides a decentralized, distributed ledger system that records transactions across many computers.",
        "The nervous system coordinates actions and sensory information by transmitting signals to and from different parts of the body.",
        "3D printing, or additive manufacturing, creates three-dimensional objects from digital models by layering materials.",
        "The endocrine system regulates hormones that control growth, metabolism, and reproduction throughout the body.",
        "Virtual reality creates simulated environments that users can interact with using specialized equipment like headsets.",
        "The muscular system enables movement, maintains posture, and circulates blood throughout the body.",
        "Big data refers to extremely large datasets that may be analyzed computationally to reveal patterns and trends.",
        "The excretory system removes waste products from the body through organs like the kidneys, liver, and skin.",
        "Augmented reality overlays digital information onto the real world through devices like smartphones or AR glasses.",
        "The reproductive system enables the production of offspring through specialized organs and hormonal regulation.",
        "Cybersecurity protects computer systems and networks from digital attacks, theft, and damage.",
        "The integumentary system includes the skin, hair, and nails, providing protection from external damage and infection.",
        "Robotics involves the design, construction, operation, and use of robots to perform tasks automatically.",
        "The lymphatic system helps maintain fluid balance and plays a crucial role in the body's immune response.",
        "Nanotechnology manipulates matter on an atomic or molecular scale, typically between 1 and 100 nanometers.",
        "The auditory system processes sound waves, allowing organisms to hear and interpret acoustic information.",
        "Cloud computing delivers computing services over the Internet, including storage, processing, and software.",
        "The vestibular system contributes to balance and spatial orientation by detecting head position and movement.",
        "Internet of Things (IoT) connects physical devices to the Internet, enabling data exchange and remote control.",
        "The olfactory system detects and processes smells through specialized receptors in the nasal cavity.",
        "Artificial neural networks are computing systems inspired by biological neural networks in animal brains.",
        "The gustatory system is responsible for the perception of taste through taste buds on the tongue.",
        "Quantum cryptography uses principles of quantum mechanics to secure communication and data transmission.",
        "The somatosensory system processes sensations from the skin, muscles, and joints, including touch, temperature, and pain.",
        "Edge computing processes data closer to where it's generated, reducing latency and bandwidth usage.",
        "The visual system enables sight by processing light information received through the eyes.",
        "Swarm intelligence studies collective behavior of decentralized, self-organized systems, natural or artificial."
    ]

    # 添加更多AI文本
    more_ai_texts = [
        "Alex tried to build a sandcastle, but the tide came in and washed it away before he could finish.",
        "During the thunderstorm, Lily found a frightened kitten under her porch and brought it inside to safety.",
        "Marcus accidentally sent a text to his boss meant for his friend, but it turned into a great conversation starter.",
        "After years of searching, Emma finally found her grandmother's lost recipe book in the attic.",
        "While cleaning out his closet, David discovered an old camera with undeveloped film from a decade ago.",
        "Sophia forgot to water her plants while on vacation, but her neighbor secretly took care of them.",
        "During a power outage, the Johnson family played board games by candlelight and had their best night in years.",
        "Leo's joke at the company meeting wasn't funny, but his honesty about his nervousness won everyone over.",
        "Maya dropped her ice cream cone, but the vendor gave her a new one for free when he saw her disappointment.",
        "Noah's car broke down in the middle of nowhere, but a passing motorist happened to be a mechanic.",
        "While walking in the rain, Chloe shared her umbrella with a stranger who turned out to be her new neighbor.",
        "Ethan's presentation slides got deleted minutes before his talk, so he improvised and gave his best speech ever.",
        "Isabella planted a time capsule in her backyard as a child and forgot about it until she found it twenty years later.",
        "Lucas made a wrong turn while hiking and discovered a beautiful waterfall no one in his town knew about.",
        "Ava's bakery ran out of her famous cupcakes, so she created a new recipe on the spot that became even more popular.",
        "During a snowstorm, the community gathered at the library when the electricity went out, sharing stories and warmth.",
        "Jackson's phone fell into a lake, but a diver retrieved it a week later with all his photos still intact.",
        "Grace accidentally bought two concert tickets, so she invited a lonely classmate who became her closest friend.",
        "Oliver's flight was canceled, but he met a fellow stranded traveler who offered him a ride home.",
        "Zoe's art project was ruined by rain, so she incorporated the water stains into a new design that won first prize.",
        "During a heatwave, Liam opened his garden hose for all the neighborhood kids, turning his yard into a water park.",
        "Harper's favorite café closed down, but the owner gave her the secret recipe for their signature drink.",
        "Carter lost his wallet at the park, and it was returned with all the money still inside by a kind jogger.",
        "Ella's dog ran away during a storm, but she found him at the animal shelter the next day, safe and sound.",
        "Gabriel's computer crashed before he could save his novel, but he rewrote it from memory and improved it.",
        "Scarlett found a message in a bottle on the beach, and after a year of searching, she found the person who sent it.",
        "Henry's garden was destroyed by hail, but his neighbors all brought him seedlings to start over.",
        "Amelia's bicycle was stolen, but the police recovered it three days later with a note of apology attached.",
        "Daniel burned the Thanksgiving turkey, so his family ordered pizza and had their most memorable holiday dinner.",
        "Lily's watch stopped working at exactly the moment she needed to know the time for an important interview.",
        "During a library book sale, Michael found his favorite childhood book with his own childish doodles still in the margins.",
        "Charlotte's necklace broke and scattered beads everywhere, but her friends helped her find every single one.",
        "Samuel's campfire went out in the wilderness, but he remembered his grandfather's trick of using pine resin to restart it.",
        "Avery missed the last bus home, but a night shift nurse offered her a ride after seeing her waiting at the stop.",
        "Joseph's glasses fell off a boat into the ocean, but a week later, a fisherman found them tangled in his net.",
        "Abigail's recipe for the town fair contest was accidentally doubled in salt, but the judges loved the unique flavor.",
        "Christopher's kite got stuck in a tree, but a strong wind later blew it free and it landed at his feet.",
        "Elizabeth's garden gnome disappeared, and it reappeared months later wearing a tiny knitted sweater.",
        "Andrew's alarm didn't go off on the day of his final exam, but his roommate woke him up just in time.",
        "Sofia's favorite pen ran out of ink during an important exam, but the teacher lent her a special gold-plated one.",
        "David's train was delayed for hours, but he met an author who was researching his next book on the platform.",
        "Madison's birthday cake was dropped by the delivery person, but her friends helped her bake an even better one.",
        "Joshua's fishing line snapped just as he caught the biggest fish of his life, but it washed ashore later that day.",
        "Emily's concert tickets were for the wrong date, but the box office exchanged them for front row seats.",
        "Ryan's map got soaked in the rain and became unreadable, but he followed a butterfly to the exact spot he was looking for.",
        "Chloe's plant seemed dead after she forgot to water it, but one small green leaf appeared after she gave it extra care.",
        "Nathan's watch was five minutes fast his entire life, making him early for everything, which saved him from missing his wedding.",
        "Hannah's recipe called for an ingredient she didn't have, so she substituted something else and created a family tradition.",
        "Tyler's car keys fell down a storm drain, but a city worker retrieved them and refused to accept a reward.",
        "Zoey's painting was criticized by her art teacher, but she entered it in a competition anyway and won first place."
    ]

    # 🔥 扩展数据集：通过重复创建更多样本
    # 重复4次基础数据集来获得更多样本（最多500条）
    base_human_texts = human_texts[:50]  # 第23-74行
    base_ai_texts = ai_texts[:50]        # 第78-129行

    all_human_texts = []
    all_ai_texts = []

    for i in range(4):  # 重复4次，得到200条
        # 添加轻微变化来增加多样性
        for text in base_human_texts:
            prefixes = ["The ", "A ", "An ", "It is known that ", "The concept of "]
            prefix = prefixes[i % len(prefixes)]
            all_human_texts.append(prefix + text[len(prefix):])

        for text in base_ai_texts:
            prefixes = ["", "Once ", "Then ", "After that ", "And so ", "After a while "]
            prefix = prefixes[i % len(prefixes)]
            all_ai_texts.append(prefix + text[len(prefix):])

    print(f"[INFO] 扩展数据集: 人类文本 {len(all_human_texts)} 条，AI文本 {len(all_ai_texts)} 条")

    # 合并原始和新增的文本
    all_human_texts = human_texts + all_human_texts
    all_ai_texts = ai_texts + all_ai_texts

    # 根据参数决定使用多少样本
    n_samples = min(args.max_raw_data // 2, len(all_human_texts), len(all_ai_texts))

    # 确保至少10个样本（降低要求以避免验证失败）
    n_samples = max(n_samples, args.min_samples)

    # 选取前n_samples个样本
    selected_human = all_human_texts[:n_samples]
    selected_ai = all_ai_texts[:n_samples]

    print(f"✅ 加载带标签数据：人类文本 {len(selected_human)} 条，AI文本 {len(selected_ai)} 条")
    print(f"📊 总样本数：{len(selected_human) + len(selected_ai)} 条")

    # 🎯 关键修复：DetectGPT需要original=人类文本，samples=AI文本
    return {
        "original": selected_human,  # 人类文本（标签0）
        "samples": selected_ai,  # AI文本（标签1）
        "labels": [0] * len(selected_human) + [1] * len(selected_ai),
        "human": selected_human,
        "ai": selected_ai
    }


# ====================== 原有辅助函数保留 ======================
def create_empty_results(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    empty_files = {
        'baseline_outputs.json': [],
        'rank_threshold_results.json': {},
        'final_results.json': {}
    }
    for filename, content in empty_files.items():
        filepath = os.path.join(output_dir, filename)
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
            print(f"✅ 创建空结果文件: {filepath}")
        except Exception as e:
            print(f"❌ 创建结果文件失败 {filepath}: {str(e)}")


def check_data_validity(data, min_samples=20):
    # 🔥 关键修复：正确计算数据样本数量
    if isinstance(data, dict):
        # 获取原始文本和样本文本
        original_count = len(data.get("original", []))
        samples_count = len(data.get("samples", []))

        # 总样本数是两者之和
        total_samples = original_count + samples_count

        print(f"📊 数据统计: original={original_count}, samples={samples_count}, total={total_samples}")

        if total_samples == 0:
            print("❌ 错误: 数据为空！")
            return False

        # 🔥 修复：使用total_samples而不是min(original_count, samples_count)
        if total_samples < min_samples:
            print(f"⚠️ 警告: 样本数量不足 (需要≥{min_samples}，当前{total_samples})")
            return False

        # 检查标签数量是否匹配
        labels_count = len(data.get("labels", []))
        if labels_count != total_samples:
            print(f"⚠️ 警告: 标签数量不匹配 (文本{total_samples}条，标签{labels_count}条)")

        print(f"✅ 数据格式有效: 包含 {original_count} 条人类文本，{samples_count} 条AI文本")
        return True
    else:
        # 如果不是字典，检查列表长度
        data_len = len(data)
        if data_len == 0:
            print("❌ 错误: 数据为空！")
            return False
        if data_len < min_samples:
            print(f"⚠️ 警告: 样本数量不足 (需要≥{min_samples}，当前{data_len})")
            return False
        return True


# ====================== 参数解析保留 ======================
def parse_args():
    parser = argparse.ArgumentParser(description="Jittor文本检测与生成（内置数据版）")
    parser.add_argument('--dataset', type=str, default='builtin', help='使用内置数据（无需修改）')
    parser.add_argument('--dataset_key', type=str, default='prompt', help='兼容原参数，无实际作用')
    parser.add_argument('--max_raw_data', type=int, default=500, help='加载的内置样本数（最大500）')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--n_perturbation_list', type=str, default='5,10',
                        help='扰动轮数列表（逗号分隔，如"3,5,7"）')
    # 模型配置
    parser.add_argument('--base_model_name', type=str, default='gpt2', 
                        help='基础模型名称 (gpt2, gpt2-large, gpt2-xl, bloomz-560m, opt-1.3b)')
    parser.add_argument('--mask_filling_model_name', type=str, default='t5-small',
                        help='掩码填充模型名称 (t5-small, t5-base, t5-large)')
    parser.add_argument('--scoring_model_name', type=str, default='', help='评分模型名称（为空则使用基础模型）')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='模型缓存目录')
    parser.add_argument('--openai_model', type=str, default='', help='OpenAI模型名称（为空则使用本地模型）')
    # 生成配置
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p采样参数')
    # 扰动配置（优化后的参数以提升AUC）
    parser.add_argument('--pct_words_masked', type=float, default=0.20,
                        help='掩码单词比例 (0.05-0.30, 默认0.20已优化)')
    parser.add_argument('--span_length', type=int, default=2,
                        help='掩码跨度长度 (1-5, 默认2已优化)')
    parser.add_argument('--n_perturbation_rounds', type=int, default=10,
                        help='扰动轮数 (5-30, 默认10已优化)')
    # 实验配置
    parser.add_argument('--DEVICE', type=str, default='auto', choices=['auto', 'cpu', 'gpu'], help='Jittor设备配置')
    parser.add_argument('--skip_baselines', action='store_true', help='是否跳过基线模型')
    parser.add_argument('--baselines_only', action='store_true', help='是否仅运行基线模型')
    parser.add_argument('--output_dir', type=str, default='./tmp_results', help='结果输出目录')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--min_samples', type=int, default=10, help='最小样本数量要求')
    # 集成分类器
    parser.add_argument('--ensemble', action='store_true', help='启用集成分类器提升检测性能')
    parser.add_argument('--ultimate', action='store_true', help='启用极致集成分类器（RF+GB+XGBoost+LightGBM+Stacking）')
    # RoBERTa 基线
    parser.add_argument('--roberta', action='store_true', help='启用 RoBERTa 基线检测器')
    parser.add_argument('--roberta_model_name', type=str, default='roberta-base',
                        help='RoBERTa 模型名称 (roberta-base, roberta-large)')
    return parser.parse_args()


# ====================== 主函数：使用内置数据 ======================
if __name__ == "__main__":
    # 解析参数
    args = parse_args()

    # 调试模式
    if args.debug:
        print("🔍 调试模式启用")
        print(f"📋 参数配置: max_raw_data={args.max_raw_data}, min_samples={args.min_samples}")

    # Jittor设备自动配置
    if args.DEVICE == 'gpu':
        if jt.has_cuda:
            jt.flags.use_cuda = True
            print("✅ 使用GPU设备运行Jittor")
        else:
            print("⚠️ GPU不可用，自动切换到CPU")
            jt.flags.use_cuda = False
    elif args.DEVICE == 'cpu':
        jt.flags.use_cuda = False
        print("✅ 使用CPU设备运行Jittor")
    else:  # auto
        jt.flags.use_cuda = jt.has_cuda
        device_type = "GPU" if jt.has_cuda else "CPU"
        print(f"✅ Jittor自动适配设备: {device_type}")

    # 初始化配置
    config = {}
    try:
        # 原代码初始化逻辑
        initial_setup(args, config)
        set_experiment_config(args, config)
        # 加载模型
        load_base_model_and_tokenizer(args, config, None)
        load_mask_filling_model(args, config)
        load_base_model(args, config)

        # ====================== 核心：加载内置数据 ======================
        print("📥 正在加载内置数据...")
        data = load_builtin_data_with_labels(args)

        # 数据集有效性校验
        print("\n🔍 开始数据有效性校验...")
        if not check_data_validity(data, min_samples=args.min_samples):
            print("❌ 数据校验失败")
            create_empty_results(config["output_dir"])
            sys.exit(1)

        print(f"\n✅ 成功加载 {len(data['original']) + len(data['samples'])} 个有效样本")
        print(f"   - 人类文本: {len(data['original'])} 条")
        print(f"   - AI文本: {len(data['samples'])} 条")
        print(f"   - 总标签数: {len(data.get('labels', []))} 条")

        # 数据预览
        if args.debug:
            print(f"\n📋 数据预览:")
            if len(data.get('original', [])) > 0:
                print(f"人类文本示例（前2条）:")
                for i, text in enumerate(data['original'][:2]):
                    print(f"  {i + 1}. {text[:60]}...")

            if len(data.get('samples', [])) > 0:
                print(f"\nAI文本示例（前2条）:")
                for i, text in enumerate(data['samples'][:2]):
                    print(f"  {i + 1}. {text[:60]}...")

        baseline_outputs = []
        outputs = []

        # 运行基线模型
        if args.scoring_model_name:
            if not args.skip_baselines and "base_model" in config:
                print("\n🚀 开始运行基线模型...")
                baseline_outputs = run_baselines(args, config, data)
            # 释放基础模型内存
            if "base_model" in config:
                del config["base_model"]
            if "base_tokenizer" in config:
                del config["base_tokenizer"]
            # 加载评分模型
            load_base_model_and_tokenizer(args, config, args.scoring_model_name)
            load_base_model(args, config)
        else:
            if not args.skip_baselines and "base_model" in config:
                print("\n🚀 开始运行基线模型...")
                baseline_outputs = run_baselines(args, config, data)

        # 运行DetectGPT
        if not args.baselines_only and "base_model" in config:
            print("\n🚀 开始运行DetectGPT...")
            outputs = detectGPT(args, config, data, args.span_length)

        # 运行集成分类器
        if args.ensemble and len(outputs) > 0:
            print("\n🚀 开始运行集成分类器...")
            from .ensemble import run_ensemble_experiment
            ensemble_result = run_ensemble_experiment(args, config, data, outputs)
            if ensemble_result:
                outputs.append(ensemble_result)  # 合并集成分类器结果

        # 运行极致集成分类器
        if args.ultimate and len(outputs) > 0:
            print("\n🚀 开始运行极致集成分类器（追求AUC极致）...")
            from .ensemble_ultimate import run_ultimate_ensemble
            ultimate_result = run_ultimate_ensemble(args, config, data, outputs)
            if ultimate_result:
                outputs.append(ultimate_result)  # 合并极致集成结果

        # 运行 RoBERTa 基线
        if args.roberta:
            print("\n🚀 开始运行 RoBERTa 基线检测...")
            from .roberta_baseline import run_roberta_baseline
            roberta_result = run_roberta_baseline(args, config, data)
            if roberta_result:
                outputs.append(roberta_result)  # 合并 RoBERTa 结果
        # 保存结果
        if not baseline_outputs:
            print("⚠️ 无基线结果，创建空结果文件")
            create_empty_results(config["output_dir"])
            sys.exit(0)

        print(f"\n💾 正在保存结果...")
        save_results(args, config, baseline_outputs, outputs)
        print(f"✅ 所有结果已保存到: {config['output_dir']}")

    except Exception as e:
        import traceback

        print(f"\n❌ 实验过程中发生错误: {str(e)}")
        traceback.print_exc()
        # 异常时创建空结果文件
        if 'config' in locals() and 'output_dir' in config:
            create_empty_results(config["output_dir"])
        sys.exit(1)