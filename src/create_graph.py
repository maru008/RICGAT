import sys
from pathlib import Path
import pandas as pd
import numpy as np
import json
from tqdm import tqdm

RI_NI_sim_ls = [0.9,0.75,0.6,0.5,0.3,0.1]
NI_C_sim = 0.75

Graph_root_dir = Path("../data/custom_graphs")

RI_NI_sim_str = "-".join(str(value).replace(".", "") for value in RI_NI_sim_ls)
folder_name = f"RINI_{RI_NI_sim_str}"
folder_path = Graph_root_dir / folder_name

# フォルダを生成
folder_path.mkdir(parents=True, exist_ok=True)
folder_log_path = folder_path / "logs"  
folder_log_path.mkdir(parents=True, exist_ok=True)

node_files = list(folder_path.glob("node_df_all_*.csv"))
edge_files = list(folder_path.glob("edge_df_all_*.csv"))

if node_files and edge_files:
    print("The graph aleady created.")
    sys.exit()
    
# ==========================================================================================
print("Start creating graph...")

node_col = ["node_id","node_name","node_type","data_source","data_source_id"]
edge_col = ["node_id1","node_id2","edge_type","data_source","node_name1","node_name2"]

node_df_all = pd.DataFrame(columns=node_col)
edge_df_all = pd.DataFrame(columns=edge_col)

now_node_id = 0


# R-RI (レシピー食材)

kikkoman_data_path = Path("/home/yoshimaru/RecipeJP/data_recipe_kikkoman/csv")

ingredient_file = kikkoman_data_path/"ingredients_kikkoman.csv"
recipe_file = kikkoman_data_path/'recipes_kikkoman.csv'

kikkoman_ingredient = pd.read_csv(ingredient_file)
kikkoman_recipe = pd.read_csv(recipe_file)

kikkoman_recipe_label = kikkoman_recipe[["recipe_id","energy","sodium","protein","lipid","dietary_fiber","sugar_content"]]

kikkoman_recipe_label.loc[:, "recipe_id"] = kikkoman_recipe_label["recipe_id"].astype(str)
kikkoman_recipe_label.loc[:, "energy"] = kikkoman_recipe_label["energy"].str.replace("kcal", "").astype(float)
kikkoman_recipe_label.loc[:, "sodium"] = kikkoman_recipe_label["sodium"].str.replace("g", "").astype(float)
kikkoman_recipe_label.loc[:, "protein"] = kikkoman_recipe_label["protein"].str.replace("g", "").astype(float)
kikkoman_recipe_label.loc[:, "lipid"] = kikkoman_recipe_label["lipid"].str.replace("g", "").astype(float)
kikkoman_recipe_label.loc[:, "dietary_fiber"] = kikkoman_recipe_label["dietary_fiber"].str.replace("g", "").astype(float)
kikkoman_recipe_label.loc[:, "sugar_content"] = kikkoman_recipe_label["sugar_content"].str.replace("g", "").astype(float)
kikkoman_recipe_label.to_csv(folder_path/"kikkoman_recipe_label.csv", index=False)


node_df_R = kikkoman_recipe[['recipe_id','title']]
node_df_R.rename(columns={'recipe_id':'data_source_id','title':'node_name'}, inplace=True)
node_df_R["data_source"] = "kikkoman"
node_df_R["node_type"] = "R"
node_df_R["node_id"] = range(now_node_id, now_node_id + len(node_df_R))
node_df_R[node_col].to_csv(folder_log_path/"01_node_df_R.csv", index=False)

node_df_R = pd.read_csv(folder_log_path/"01_node_df_R.csv")
now_node_id = max(node_df_R["node_id"]) + 1
node_df_all = pd.concat([node_df_all, node_df_R], axis=0)

node_df_RI = kikkoman_ingredient[["ingredient_id","name"]].drop_duplicates()
node_df_RI.rename(columns={'ingredient_id':'data_source_id','name':'node_name'}, inplace=True)
node_df_RI["node_type"] = "RI"
node_df_RI["data_source"] = "kikkoman"
node_df_RI["node_id"] = range(now_node_id, now_node_id + len(node_df_RI))
node_df_RI[node_col].to_csv(folder_log_path/"02_node_df_RI.csv", index=False)

node_df_RI = pd.read_csv(folder_log_path/"02_node_df_RI.csv")
now_node_id = max(node_df_RI["node_id"]) + 1
node_df_all = pd.concat([node_df_all, node_df_RI], axis=0)


edge_ls_R_RI = []
edge_col = ["node_id1","node_id2","edge_type","data_source","node_name1","node_name2"]

recipe_id2node_id = dict(zip(node_df_R["data_source_id"], node_df_R["node_id"]))
ingredient_id2node_id = dict(zip(node_df_RI["data_source_id"], node_df_RI["node_id"]))
recipe_id2node_name = dict(zip(node_df_R["data_source_id"], node_df_R["node_name"]))
ingredient_id2node_name = dict(zip(node_df_RI["data_source_id"], node_df_RI["node_name"]))

for i, row in tqdm(kikkoman_ingredient.iterrows(), total=len(kikkoman_recipe), desc="R-RI edge"):
    recipe_id = row["recipe_id"]
    ingred_id = row["ingredient_id"]
    add_ls = []
    add_ls.append(recipe_id2node_id[recipe_id])
    add_ls.append(ingredient_id2node_id[ingred_id])
    add_ls.append("R-RI")
    add_ls.append("kikkoman")
    add_ls.append(recipe_id2node_name[recipe_id])
    add_ls.append(ingredient_id2node_name[ingred_id])
    edge_ls_R_RI.append(add_ls)
edge_df_R_RI = pd.DataFrame(edge_ls_R_RI, columns=edge_col)
edge_df_R_RI.to_csv(folder_log_path/"03_edge_df_R_RI.csv", index=False)

edge_df_R_RI = pd.read_csv(folder_log_path/"03_edge_df_R_RI.csv")
edge_df_all = pd.concat([edge_df_all, edge_df_R_RI], axis=0)

# NI-NI (成分表食材同士)

# 成分表食材読み込み
Foodtable_data_path = Path("/mnt/d/data_nutrient_FoodTableJP/data_preprocessed")
Foodtable_ingri_file = Foodtable_data_path/"food_categories_all.csv"
Foodtable_nutri_file = Foodtable_data_path/"food_nutrition_all.csv"
Foodtable_ingredient = pd.read_csv(Foodtable_ingri_file)
Foodtable_nutirition = pd.read_csv(Foodtable_nutri_file)


def add_supposition_nodes(df):
    df = df.fillna("")
    new_rows = []
    for index, row in df.iterrows():
        base_food_number = row["食品番号"]
        
        if row["副分類"]:
            # 副分類のみ
            new_rows.append({
                "食品番号": "supposition_node",
                "食品名": row["副分類"],
                "副分類": row["副分類"],
                "副分類": "",
                "大分類": "",
                "中分類": "",
                "小分類": "",
                "細分": "",
                "階層": 1
            })

        if row["大分類"]:
            # 副分類＋大分類
            if row["副分類"]:
                new_rows.append({
                    "食品番号": "supposition_node",
                    "食品名": f"{row['副分類']}　{row['大分類']}",
                    "副分類": row["副分類"],
                    "副分類": "",
                    "大分類": row["大分類"],
                    "中分類": "",
                    "小分類": "",
                    "細分": "",
                    "階層": 2
                })
            # 大分類のみ
            else:
                new_rows.append({
                    "食品番号": "supposition_node",
                    "食品名": row["大分類"],
                    "副分類": "",
                    "副分類": "",
                    "大分類": row["大分類"],
                    "中分類": "",
                    "小分類": "",
                    "細分": "",
                    "階層": 1
                })
            
            # 大分類＋中分類
            if row["中分類"]:
                new_rows.append({
                    "食品番号": "supposition_node",
                    "食品名": f"{row['大分類']}　{row['中分類']}",
                    "副分類": row["副分類"] if row["副分類"] else "",
                    "副分類": "",
                    "大分類": row["大分類"],
                    "中分類": row["中分類"],
                    "小分類": "",
                    "細分": "",
                    "階層": 3
                })

            # 大分類＋中分類＋小分類
            if row["小分類"]:
                new_rows.append({
                    "食品番号": "supposition_node",
                    "食品名": f"{row['大分類']}　{row['中分類']}　{row['小分類']}",
                    "副分類": row["副分類"] if row["副分類"] else "",
                    "副分類": "",
                    "大分類": row["大分類"],
                    "中分類": row["中分類"] if row["中分類"] else "",
                    "小分類": row["小分類"],
                    "細分": "",
                    "階層": 4
                })

            # 大分類＋中分類＋小分類＋細分
            if row["細分"]:
                new_rows.append({
                    "食品番号": "supposition_node",
                    "食品名": f"{row['大分類']}　{row['中分類']}　{row['小分類']}　{row['細分']}",
                    "副分類": row["副分類"] if row["副分類"] else "",
                    "副分類": "",
                    "大分類": row["大分類"],
                    "中分類": row["中分類"] if row["中分類"] else "",
                    "小分類": row["小分類"],
                    "細分": row["細分"],
                    "階層": 5
                })
    return pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

# 新しいデータフレームの作成
Foodtable_ingredient["階層"] = 0
new_df = add_supposition_nodes(Foodtable_ingredient).drop_duplicates()
new_df['食品名'] = new_df['食品名'].str.replace('　　', '　')
new_df = new_df.drop_duplicates()
#現在のノードID定義
now_node_id = max(node_df_RI['node_id']) + 1
# ノードID作成
new_df['node_id'] = range(now_node_id, now_node_id + len(new_df))
# ノードデータフレームを作成
node_df_foodTable = new_df[['node_id', '食品名', '副分類', '食品番号',"階層"]].copy()
node_df_foodTable.columns = ['node_id', 'node_name', 'node_type', 'data_source_id','tree_level']
node_df_foodTable['node_type'] = 'NI'
node_df_foodTable['data_source'] = 'FoodTable'
name_to_id = {row['node_name']: row['node_id'] for _, row in node_df_foodTable.iterrows()}
node_col.append('tree_level')
node_df_NI = node_df_foodTable.copy()[node_col]
node_df_NI.to_csv(folder_log_path/"04_node_df_NI.csv", index=False)

node_df_NI = pd.read_csv(folder_log_path/"04_node_df_NI.csv")
node_df_all = pd.concat([node_df_all, node_df_NI], axis=0)

def create_edge_df(df):
    edges = []
    for _, row in df.iterrows():
        current_name = row['食品名']
        current_id = row['node_id']
        
        # 細分 -> 小分類
        if pd.notna(row['細分']):
            parent_name = f"{row['大分類']}　{row['中分類']}　{row['小分類']}"
            if parent_name in name_to_id:
                edges.append([name_to_id[parent_name], current_id, 'NI-NI', 'FoodTable'])
                continue
        
        # 小分類 -> 中分類
        if pd.notna(row['小分類']):
            parent_name = f"{row['大分類']}　{row['中分類']}"
            if parent_name in name_to_id:
                edges.append([name_to_id[parent_name], current_id, 'NI-NI', 'FoodTable'])
                continue
        
        # 中分類 -> 大分類
        if pd.notna(row['中分類']):
            parent_name = row['大分類']
            if parent_name in name_to_id:
                edges.append([name_to_id[parent_name], current_id, 'NI-NI', 'FoodTable'])
                continue
        
        # 大分類 -> 類区分
        if pd.notna(row['大分類']):
            parent_name = row['類区分']
            if parent_name in name_to_id:
                edges.append([name_to_id[parent_name], current_id, 'NI-NI', 'FoodTable'])
    
    edge_df = pd.DataFrame(edges, columns=["node_id1", "node_id2", "edge_type", "data_source"])
    return edge_df

# エッジデータフレームを作成
edge_df_NI_NI = create_edge_df(new_df)
node_id2name = {row['node_id']: row['node_name'] for _, row in node_df_NI.iterrows()}


edge_df_NI_NI['node_name1'] = edge_df_NI_NI['node_id1'].map(node_id2name)
edge_df_NI_NI['node_name2'] = edge_df_NI_NI['node_id2'].map(node_id2name)
node_id2level = {row['node_id']: row['tree_level'] for _, row in node_df_NI.iterrows()}
edge_df_NI_NI["levels"] = edge_df_NI_NI["node_id1"].map(node_id2level).astype(str) + "-" + edge_df_NI_NI["node_id2"].map(node_id2level).astype(str)


edge_col.append("levels")
edge_df_NI_NI[edge_col].to_csv(folder_log_path/"05_edge_df_NI_NI.csv", index=False)

edge_df_NI_NI = pd.read_csv(folder_log_path/"05_edge_df_NI_NI.csv")
edge_df_all = pd.concat([edge_df_all, edge_df_NI_NI], axis=0)

# RI-NI　(レシピ食材 -- 成分表食材)
# kikkoman ingredient読み込み
kikkoman_data_path = Path("/home/yoshimaru/RecipeJP/data_recipe_kikkoman/csv")

ingredient_file = kikkoman_data_path/"ingredients_kikkoman.csv"
recipe_file = kikkoman_data_path/'recipes_kikkoman.csv'
steps_file = kikkoman_data_path/'steps_kikkoman.csv'
tags_file = kikkoman_data_path/'tags_kikkoman.csv'
kikkoman_embedding_file = "/mnt/d/data_embeddings/Kikkoman/OpenAI_embedding_3_large/ingredients/kikkoman_ingredient_embed.json"

# kikkoman_ingredient = pd.read_csv(ingredient_file)
# kikkoman_recipe = pd.read_csv(recipe_file)

# kikkoman_ingredientのembedding読み込み
with open(kikkoman_embedding_file, "r") as f:
    kikkoman_ingredient_embed = json.load(f)
    
# 成分表 embed 
foodtable_embed_root_dir = Path("/mnt/d/data_embeddings/FoodTableJP/OpenAI_embedding_3_large/ingredients")
foodtable_embed_save_path = foodtable_embed_root_dir / "foodtable_ingredients.json"
with open(foodtable_embed_save_path, "r") as f:
    foodtable_ingredient_embed = json.load(f)
    
    
def cosine_similarity_matrix(v1, v2_matrix):
    """
    ベクトルv1と行列v2_matrixの各行とのコサイン類似度を計算する
    """
    v1_norm = np.linalg.norm(v1)
    v2_norms = np.linalg.norm(v2_matrix, axis=1)
    dot_products = np.dot(v2_matrix, v1)
    return dot_products / (v1_norm * v2_norms)

def search_cos_max_node(trg_df, trg_embed):
    """
    cosine類似度が最大となるノードを探す
    """
    node_names = trg_df["node_name"].values
    node_ids = trg_df["node_id"].values
    node_embeds = np.array([foodtable_ingredient_embed[name] for name in node_names])
    
    cos_scores = cosine_similarity_matrix(trg_embed, node_embeds)
    max_sim_index = np.argmax(cos_scores)
    max_sim_score = cos_scores[max_sim_index]
    max_sim_node_id = node_ids[max_sim_index]
    
    return max_sim_node_id, max_sim_score

def judge_edge_RI_NI(kikkoman_ingredient_id, RI_NI_sim_ls):
    """
    コサイン類似度が最大を特定し，エッジを作成する
    """
    trg_embed = kikkoman_ingredient_embed[str(kikkoman_ingredient_id)]
    for level in range(len(RI_NI_sim_ls)):
        foodtable_node_i = node_df_NI[node_df_NI["tree_level"] == level]
        if foodtable_node_i.empty:
            continue
        res_node_id, sim_score = search_cos_max_node(foodtable_node_i, trg_embed)
        if sim_score >= RI_NI_sim_ls[level]:
            return res_node_id, sim_score, level
    return None, None, None


#マルチコア処理
from multiprocessing import Pool, freeze_support, RLock

def process_row(args):
    index, row = args
    trg_node_id = row["node_id"]
    kikkoman_ingredient_id = row["data_source_id"]
    res_node_id, sim_score, connected_level = judge_edge_RI_NI(kikkoman_ingredient_id, RI_NI_sim_ls)
    
    if res_node_id:
        return [trg_node_id, res_node_id, "RI-NI", "kikkoman&foodtable", row["node_name"], node_id2name[res_node_id], f"RI-{connected_level}"]
    return None


def parallel_process(data):
    tqdm.set_lock(RLock())  # 必要なロック設定
    with Pool(initializer=tqdm.set_lock, initargs=(RLock(),)) as pool:
        results = list(tqdm(pool.imap(process_row, data.iterrows()), total=len(data), desc="Processing", position=0))
    return results


node_id2name = {row['node_id']: row['node_name'] for _, row in node_df_all.iterrows()}
    
results = parallel_process(node_df_RI)
    
edge_data = [res for res in results if res is not None]
edge_df_RI_NI = pd.DataFrame(edge_data, columns=["node_id1", "node_id2", "edge_type", "data_source", "node_name1", "node_name2", "levels"])


#統計データを保存
gby_edges = edge_df_RI_NI.groupby("levels").count().reset_index()[["levels","node_id1"]].rename(columns={"node_id1":"count"})
total = pd.DataFrame([["total", sum(gby_edges["count"])]], columns=["levels","count"])
all_nodes = pd.DataFrame([["all_nodes", len(kikkoman_ingredient)]], columns=["levels","count"])
gby_edges = pd.concat([gby_edges, total], axis=0)
gby_edges.to_csv(folder_log_path/"RINI_gby_edges.csv", index=False)

edge_df_RI_NI[edge_col].to_csv(folder_log_path/"06_edge_df_RI_NI.csv", index=False)

edge_df_RI_NI = pd.read_csv(folder_log_path/"06_edge_df_RI_NI.csv")
edge_df_all = pd.concat([edge_df_all, edge_df_RI_NI], axis=0)


Foodtable_data_path = Path("/mnt/d/data_nutrient_FoodTableJP/data_preprocessed")
Foodtable_ingri_file = Foodtable_data_path/"food_categories_all.csv"
Foodtable_nutri_file = Foodtable_data_path/"food_nutrition_all.csv"
Foodtable_ingredient = pd.read_csv(Foodtable_ingri_file)
Foodtable_nutirition = pd.read_csv(Foodtable_nutri_file)
Foodtable_nutirition.columns = Foodtable_nutirition.columns.str.strip()


trg_nutrition = pd.read_csv("/home/yoshimaru/FoodTableJP/FoodTableJP-preprocessing/data/annotated/food_nutrition_identifier.csv")
trg_nutrition.columns = trg_nutrition.columns.str.strip()
trg_nutrition_ls = ["food_code"]
trg_nutrition_ls.extend(trg_nutrition.dropna(subset=["Pubchem CID"])["識別子"].tolist())

nutri_id2name = {row["識別子"]: row["説明"] for _, row in trg_nutrition.iterrows()}


more_nutrition_ls = [
    "F15D1",
    "F17D1",
    "F20D3N3",
    "F20D4N3",
    "F20D4N6",
    "F20D5N3",
    "F21D5N3",
    "F22D2"
]

trg_nutrition_ls.extend(more_nutrition_ls)

import math
# trg_nutrition_lsからNaNを削除
trg_nutrition_ls_filtered = [str(x) for x in trg_nutrition_ls if not (isinstance(x, float) and math.isnan(x))]


filtered_foodtable_nutirition = Foodtable_nutirition[trg_nutrition_ls_filtered]

filtered_foodtable_nutirition.fillna(0, inplace=True)
filtered_foodtable_nutirition.replace(['Tr', '', '-'], 0, inplace=True)
# 括弧付きの数値から括弧を外す
def clean_numeric(value):
    try:
        return float(''.join(filter(lambda x: x.isdigit() or x == '.', str(value))))
    except ValueError:
        return 0

# データフレーム全体に適用
filtered_foodtable_nutirition = filtered_foodtable_nutirition.applymap(clean_numeric)


for column in filtered_foodtable_nutirition.columns[1:]:
    filtered_foodtable_nutirition[column] = filtered_foodtable_nutirition[column].astype(float)
    
# 出力用のリストを初期化
edges = []

# 各栄養素のアルファに基づく値を計算
for column in tqdm(filtered_foodtable_nutirition.columns[1:], desc='Processing',total=len(filtered_foodtable_nutirition.columns[1:])):
    # アルファに基づいて閾値を設定
    threshold = filtered_foodtable_nutirition[column].quantile(NI_C_sim)
    
    # 基準を満たす場合にエッジを追加
    for index, row in filtered_foodtable_nutirition.iterrows():
        if row[column] >= threshold:
            edges.append({'Food_code': row['food_code'], 'Nutri': column})

# 結果をデータフレームに変換
edge_df_NI_C_onlyID = pd.DataFrame(edges)
NI_C_sim_int = int(NI_C_sim * 100)
edge_df_NI_C_onlyID["Food_code"] = edge_df_NI_C_onlyID["Food_code"].astype(int)

print("エッジ数：",len(edge_df_NI_C_onlyID))
print("エッジが貼られた割合：",round((len(edge_df_NI_C_onlyID)/(2541*184) )*100,1),"%")


now_node_id = max(node_df_NI["node_id"]) + 1
node_ls_NI_C = []
for nutri_i in list(edge_df_NI_C_onlyID["Nutri"].unique()):
    node_ls_NI_C.append([nutri_id2name[nutri_i],"C","FoodTable",nutri_i])
node_df_C = pd.DataFrame(node_ls_NI_C,columns=["node_name","node_type","data_source","data_source_id"])
node_df_C["node_id"] = range(now_node_id, now_node_id + len(node_df_C))
node_df_C["tree_level"] = np.nan
node_df_C[node_col].to_csv(folder_log_path/f"07_node_df_C{NI_C_sim_int}.csv", index=False)

node_df_C = pd.read_csv(folder_log_path/f"07_node_df_C{NI_C_sim_int}.csv")
node_df_all = pd.concat([node_df_all, node_df_C], axis=0)

C_sourceid2nodeid = {row["data_source_id"]: row["node_id"] for _, row in node_df_C.iterrows()}
C_sourceid2name = {row["data_source_id"]: row["node_name"] for _, row in node_df_C.iterrows()}
NI_sourceid2nodeid = {row["data_source_id"]: row["node_id"] for _, row in node_df_NI.iterrows()}
NI_sourceid2name = {row["data_source_id"]: row["node_name"] for _, row in node_df_NI.iterrows()}


edge_ls_NI_C = []
for index, row in tqdm(edge_df_NI_C_onlyID.iterrows(), total=len(edge_df_NI_C_onlyID), desc="NI-C edge"):
    NI_source_id = str(row["Food_code"])
    C_source_id = row["Nutri"]
    add_edge = [NI_sourceid2nodeid[NI_source_id],C_sourceid2nodeid[C_source_id],"NI-C","FoodTable",NI_sourceid2name[NI_source_id],C_sourceid2name[C_source_id],np.nan]
    edge_ls_NI_C.append(add_edge)
edge_df_NI_C = pd.DataFrame(edge_ls_NI_C, columns=edge_col)    

edge_df_NI_C.to_csv(folder_log_path/f"08_edge_NI_C_{NI_C_sim_int}.csv", index=False)

edge_df_NI_C = pd.read_csv(folder_log_path/f"08_edge_NI_C_{NI_C_sim_int}.csv")
edge_df_all = pd.concat([edge_df_all, edge_df_NI_C], axis=0)

node_df_all = node_df_all.drop_duplicates()
edge_df_all = edge_df_all.drop_duplicates()

node_df_all.to_csv(folder_path/f"node_df_all_{len(node_df_all)}.csv", index=False)
edge_df_all.to_csv(folder_path/f"edge_df_all_{len(edge_df_all)}.csv", index=False)

import networkx as nx

G = nx.Graph()

# ノードの追加
for _, row in tqdm(node_df_all.iterrows(),total=len(node_df_all),desc="add node"):
    G.add_node(row['node_id'], name=row['node_name'], type=row['node_type'], data_source=row['data_source'], data_source_id=row['data_source_id'], tree_level=row['tree_level'])

# エッジの追加
for _, row in tqdm(edge_df_all.iterrows(),total=len(edge_df_all),desc="add edge"):
    G.add_edge(row['node_id1'], row['node_id2'], edge_type=row['edge_type'], data_source=row['data_source'])

# Edge List形式で保存
nx.write_edgelist(G, folder_path/"graph.edgelist", data=True)
# GraphML形式で保存
nx.write_graphml(G, folder_path/"graph.graphml")