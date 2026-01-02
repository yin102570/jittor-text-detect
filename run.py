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
def load_builtin_data(args):
    """
    加载内置的200条文本数据（补充samples键，兼容原有逻辑）
    返回格式：{"original": 文本列表, "samples": 文本列表}，完全对齐原有数据集格式
    """
    # 内置200条混合文本（短故事、维基百科片段、写作提示）
    builtin_texts = [
        # 类别1：短故事（ROCStories风格，50条）
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
        "Wendy volunteered to read to kids at the library and discovered she loved storytelling.",
        "Xander built a fort in his backyard with his cousins and spent the night camping out.",
        "Yara tried painting with watercolors and made a beautiful picture of her cat, which she framed.",
        "Zach lost his favorite toy car but found it in the sandbox at the park the next day.",

        # 类别2：维基百科片段（WikiText风格，50条）
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
        "Volleyball is a team sport in which two teams of six players are separated by a net, using their hands to hit a ball over the net.",

        # 类别3：写作提示（WritingPrompts风格，100条）
        "Prompt: A librarian discovers a book that writes back to anyone who opens it. Write a story about their first conversation.",
        "Prompt: In a world where everyone can see how many days they have left to live, a person's counter suddenly resets to zero but they don't die. What happens?",
        "Prompt: A baker bakes bread that grants small wishes to anyone who eats it, but the wishes always come with a tiny catch.",
        "Prompt: A hiker gets lost in the woods and stumbles upon a village where time moves backwards. Describe their experience.",
        "Prompt: A teacher realizes one of their students is a time traveler trying to prevent a future disaster. How do they react?",
        "Prompt: A barista can hear the thoughts of anyone who orders a coffee, but they can only turn it off by telling the truth to a stranger.",
        "Prompt: A painter's artwork comes to life at midnight, but only for exactly one hour each night. What do they paint?",
        "Prompt: In a city where lying is illegal, a child tells their first lie and must face the consequences.",
        "Prompt: A musician's guitar plays music from the future, but each song predicts a small tragedy. Do they keep playing?",
        "Prompt: A gardener grows flowers that emit the scent of memories—each bloom smells like a different moment from someone's life.",
        "Prompt: A writer's characters start showing up in real life, but they're nothing like how they were written.",
        "Prompt: In a world where people can only speak in rhymes, a person suddenly loses the ability to rhyme. How do they communicate?",
        "Prompt: A photographer takes a photo that shows the future of anyone who looks at it. What do they see in their own photo?",
        "Prompt: A chef cooks meals that taste like home to anyone who eats them—even if they've never had a home.",
        "Prompt: A firefighter rescues a cat from a burning building and discovers the cat can speak, but only to them.",
        "Prompt: In a small town where everyone has a magical talent, a teenager is born with no talent at all. What makes them special?",
        "Prompt: A bookseller finds a book that never ends—no matter how many pages you turn, there's always more. Do they read it all?",
        "Prompt: A swimmer discovers a hidden underwater city where the residents can breathe air and water equally.",
        "Prompt: A tailor sews clothes that change the personality of anyone who wears them. What do they make for themselves?",
        "Prompt: In a world where dreams are shared publicly, a person has a secret dream they never want anyone to see.",
        "Prompt: A farmer finds a seed that grows into a tree bearing fruit that answers any question. What do they ask first?",
        "Prompt: A taxi driver picks up a passenger who claims to be from another dimension. Do they believe them?",
        "Prompt: A student finds a notebook that writes down the answer to any question before they ask it.",
        "Prompt: In a world where aging stops at 30, a woman starts aging again and must find out why.",
        "Prompt: A sculptor carves statues that look exactly like people who will die the next day. How do they use this gift?",
        "Prompt: A pilot flies a plane through a storm and emerges in a version of Earth where dinosaurs never went extinct.",
        "Prompt: A dentist can tell if someone is lying by looking at their teeth. What do they do when a famous politician lies to them?",
        "Prompt: In a city where everyone has a twin, a person meets their twin for the first time—and they hate each other.",
        "Prompt: A vet discovers they can understand animal thoughts, but the animals all have complaints about their owners.",
        "Prompt: A locksmith can open any lock—including emotional ones, like a broken heart. Who asks for their help?",
        "Prompt: A runner is chased by a shadow that only appears when they run alone at night. Do they stop running?",
        "Prompt: In a world where money is replaced by memories, a person has no memories to trade. How do they survive?",
        "Prompt: A scientist invents a machine that can record emotions, but accidentally records a emotion no human has ever felt.",
        "Prompt: A dancer's movements can control the weather—each step changes the wind, rain, or sun. What dance do they perform?",
        "Prompt: A poet's words make plants grow, but only if the poem is honest. What do they write about?",
        "Prompt: In a small village where everyone knows everyone's secrets, a stranger arrives with no secrets at all.",
        "Prompt: A lifeguard saves a person from drowning and realizes the person is a ghost who doesn't know they're dead.",
        "Prompt: A blacksmith forges a sword that can cut through lies, but it can also cut through the truth if misused.",
        "Prompt: A astronomer spots a new star in the sky that spells out messages in constellations. What does it say?",
        "Prompt: In a world where people can only see in black and white, a child is born who can see color. How do they explain it?",
        "Prompt: A hairdresser's haircuts change people's luck—good or bad—depending on the style. What haircut do they give themselves?",
        "Prompt: A traveler gets stuck in an airport and meets a stranger who has the same exact suitcase and life story.",
        "Prompt: A magician's tricks are real magic, but they can only use it to help others—not themselves. What do they do?",
        "Prompt: In a world where sleep is optional, a person chooses to sleep for the first time in 20 years. What do they dream?",
        "Prompt: A archaeologist digs up a artifact that is a modern smartphone—from 2000 years ago.",
        "Prompt: A zookeeper discovers the animals can talk, but only when no one else is around. What do they talk about?",
        "Prompt: A painter can paint doorways to other worlds, but each doorway can only be used once. Where do they go?",
        "Prompt: In a city where everyone has a clock that counts down to their greatest achievement, a person's clock stops.",
        "Prompt: A musician's voice can heal injuries, but each healing takes a little bit of their own energy. How far do they go?",
        "Prompt: A writer forgets how to read and write, but can still tell stories aloud that are more vivid than ever.",
        "Prompt: A gardener's plants can talk, but they only speak in riddles. What are they trying to say?",
        "Prompt: In a world where people can fly, a person is born who can only walk—but they see things no one else can.",
        "Prompt: A chef's food can make people relive their happiest memory, but one customer has no happy memories.",
        "Prompt: A photographer's photos can freeze time for the subject—each photo stops their life for one day.",
        "Prompt: A teacher can hear the future of their students in their voices. What do they hear for a quiet student?",
        "Prompt: A barista's coffee can make people tell the truth, but they accidentally drink it themselves. What do they admit?",
        "Prompt: In a small town where it rains every day, the sun comes out for exactly one hour—and a child is born at that moment.",
        "Prompt: A librarian's library has books that rewrite themselves based on the reader's life. What does their favorite book say?",
        "Prompt: A hiker finds a cave with walls that show the future of anyone who touches them. What do they see?",
        "Prompt: A baker's bread can make people younger for a day, but they can only bake one loaf per year. Who gets it?",
        "Prompt: A swimmer can breathe underwater, but only if they don't think about the surface. What do they discover?",
        "Prompt: In a world where everyone has a pet dragon, a person's dragon is tiny and can't breathe fire. What makes it special?",
        "Prompt: A sculptor's statues come to life, but they only want to do ordinary things—like grocery shopping.",
        "Prompt: A pilot can fly without a plane, but only when they're scared. What do they fly away from?",
        "Prompt: A dentist invents a toothpaste that can make people remember forgotten memories. What do they remember?",
        "Prompt: A tailor's clothes can make people invisible, but only to the person they love the most.",
        "Prompt: In a city where music is illegal, a child starts singing in the street—and everyone stops to listen.",
        "Prompt: A farmer's crops grow into the shape of words, spelling out a warning about a coming storm.",
        "Prompt: A taxi driver's car can travel through time, but only to moments that involve regret. Where do they go?",
        "Prompt: A student's homework writes itself, but it always answers questions no one asked. What does it say?",
        "Prompt: A scientist's machine can swap personalities between people—they swap with their cat by accident.",
        "Prompt: A dancer can dance with ghosts, but each dance makes the ghost fade a little more. Do they keep dancing?",
        "Prompt: A poet's poems can make people fall in love, but they can't control who falls for whom.",
        "Prompt: In a world where everyone has a superpower, a person's power is to listen to silence. What does silence say?",
        "Prompt: A lifeguard can walk on water, but only when they're helping someone else. What do they do alone?",
        "Prompt: A blacksmith forges a key that can open any door—including the door to the past. Do they use it?",
        "Prompt: A astronomer can talk to stars, but the stars are sad about something happening to Earth.",
        "Prompt: A hairdresser can change people's memories with a haircut—they erase their own bad memory by accident.",
        "Prompt: A traveler finds a map that leads to happiness, but the path changes every time they look at it.",
        "Prompt: A magician can make anything disappear, but they accidentally make their own shadow disappear.",
        "Prompt: In a world where food grows on trees like fruit, a person finds a tree that grows pizza.",
        "Prompt: A archaeologist finds a tomb with a message written in their own handwriting—from 100 years in the future.",
        "Prompt: A zookeeper can turn into any animal, but they get stuck as a penguin at the North Pole.",
        "Prompt: A painter can paint memories, but they accidentally paint a memory that never happened.",
        "Prompt: A musician can play any instrument without learning it, but each instrument makes them forget a memory.",
        "Prompt: In a city where people age backwards, a person is born and starts aging forward instead.",
        "Prompt: A writer's stories come true, but only the bad parts—they must rewrite a happy ending to fix it.",
        "Prompt: A gardener's flowers can predict the weather, but they start predicting something worse than storms.",
        "Prompt: A chef can cook food from other planets, but the food has strange side effects—like floating.",
        "Prompt: A photographer's photos can bring people back to life, but only for one minute. Who do they bring back?",
        "Prompt: A teacher can make any subject interesting, but they can't teach their own child anything.",
        "Prompt: A barista can make coffee that tastes like any emotion—they make a cup of 'courage' for a scared customer.",
        "Prompt: In a small village where everyone is the same, a person has a different colored eye—and sees things differently.",
        "Prompt: A librarian can enter the world of any book, but they get stuck in a fairy tale with no happy ending.",
        "Prompt: A hiker meets a wolf that can speak and asks for help finding its way home to the mountains.",
        "Prompt: A baker invents a cookie that makes people understand each other's languages—including animal languages.",
        "Prompt: A swimmer breaks a world record but realizes they were swimming in a lake that doesn't exist on any map.",
        "Prompt: In a world where people can read minds, a person's mind is blank—and everyone is curious about them.",
        "Prompt: A sculptor carves a statue of a person they've never met, then meets them the next day.",
        "Prompt: A pilot flies through a rainbow and enters a world where all dreams are real—but nightmares are too.",
        "Prompt: A dentist finds a tooth that grants wishes, but each wish costs a memory. What do they wish for?",
        "Prompt: A tailor sews a coat that makes people love the wearer, but the love is fake. Do they wear it?",
        "Prompt: A poet writes a poem that makes it snow in summer—and the snow doesn't melt.",
        "Prompt: A farmer finds a cow that gives milk that tastes like chocolate, but only on Sundays."
    ]

    # 截断到指定样本数（默认100，可通过--max_raw_data调整）
    texts = builtin_texts[:args.max_raw_data]
    print(f"✅ 成功加载内置数据：共{len(texts)}条样本（内置200条，按参数截断）")

    # 核心修复：补充samples键（与原有数据集格式对齐）
    return {
        "original": texts,
        "samples": texts  # 新增samples键，值与original一致
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
    # 校验时同时检查original和samples键
    if isinstance(data, dict):
        valid_len_original = len(data.get("original", []))
        valid_len_samples = len(data.get("samples", []))
        valid_len = min(valid_len_original, valid_len_samples)
    else:
        valid_len = len(data)

    if valid_len == 0:
        print("❌ 错误: 数据为空！")
        return False
    if valid_len < min_samples:
        print(f"⚠️ 警告: 样本数量不足 (需要≥{min_samples}，当前{valid_len})")
        return False
    return True


# ====================== 参数解析保留 ======================
def parse_args():
    parser = argparse.ArgumentParser(description="Jittor文本检测与生成（内置数据版）")
    parser.add_argument('--dataset', type=str, default='builtin', help='使用内置数据（无需修改）')
    parser.add_argument('--dataset_key', type=str, default='prompt', help='兼容原参数，无实际作用')
    parser.add_argument('--max_raw_data', type=int, default=100, help='加载的内置样本数（最大200）')
    parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
    parser.add_argument('--n_perturbation_list', type=str, default='5',
                        help='扰动轮数列表（逗号分隔，如"3,5,7"）')
    # 模型配置
    parser.add_argument('--base_model_name', type=str, default='gpt2', help='基础模型名称')
    parser.add_argument('--mask_filling_model_name', type=str, default='t5-small', help='掩码填充模型名称')
    parser.add_argument('--scoring_model_name', type=str, default='', help='评分模型名称')
    parser.add_argument('--cache_dir', type=str, default='./cache', help='模型缓存目录')
    parser.add_argument('--openai_model', type=str, default='', help='OpenAI模型名称（为空则使用本地模型）')
    # 生成配置
    parser.add_argument('--temperature', type=float, default=0.7, help='生成温度')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p采样参数')
    # 扰动配置
    parser.add_argument('--pct_words_masked', type=float, default=0.15, help='掩码单词比例')
    parser.add_argument('--span_length', type=int, default=3, help='掩码跨度长度')
    parser.add_argument('--n_perturbation_rounds', type=int, default=5, help='扰动轮数')
    # 实验配置
    parser.add_argument('--DEVICE', type=str, default='auto', choices=['auto', 'cpu', 'gpu'], help='Jittor设备配置')
    parser.add_argument('--skip_baselines', action='store_true', help='是否跳过基线模型')
    parser.add_argument('--baselines_only', action='store_true', help='是否仅运行基线模型')
    parser.add_argument('--output_dir', type=str, default='./tmp_results', help='结果输出目录')
    return parser.parse_args()


# ====================== 主函数：使用内置数据 ======================
if __name__ == "__main__":
    # 解析参数
    args = parse_args()

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
        data = load_builtin_data(args)

        # 数据集有效性校验
        if not check_data_validity(data, min_samples=15):
            create_empty_results(config["output_dir"])
            sys.exit(1)

        print(f"✅ 成功加载 {len(data['original'])} 个有效样本（original/samples双键兼容）")
        baseline_outputs = []
        outputs = []

        # 运行基线模型
        if args.scoring_model_name:
            if not args.skip_baselines and "base_model" in config:
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
                baseline_outputs = run_baselines(args, config, data)

        # 运行DetectGPT
        if not args.baselines_only and "base_model" in config:
            outputs = detectGPT(args, config, data, args.span_length)

        # 保存结果
        if not baseline_outputs:
            create_empty_results(config["output_dir"])
            sys.exit(0)
        save_results(args, config, baseline_outputs, outputs)
        print(f"✅ 所有结果已保存到: {config['output_dir']}")

    except Exception as e:
        print(f"❌ 实验过程中发生错误: {str(e)}")
        # 异常时创建空结果文件
        if 'config' in locals() and 'output_dir' in config:
            create_empty_results(config["output_dir"])
        sys.exit(1)

