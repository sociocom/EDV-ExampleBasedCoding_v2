
import argparse
import pandas as pd
import numpy as np
import pprint
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import yaml
from types import SimpleNamespace
import os

from ExampleBasedCoding import EntityDictionary, normalize
from VectorBasedCoding import EntityDictionary as VectorEntityDictionary
from VectorBasedCoding import normalize as vector_normalize



"""
Config arguments:

    Path
        input: Input Excel file path
        output: Output Excel file path
    
    Column names
        id: ID column in Excel file
        source: Source column name in Excel file
        target: Target column name in Excel file
        flag_col: Flag column name in Excel file
    
    Flag list
        source_flag: Source flag list for training data. If the flags are more than one, split by comma without space. nan is for the cells missing values. (Ex: [A, B, C]) 
        target_flag: Target flag list for test data. If the flags are more than one, Split by comma without space. nan is for the cells missing values. (Ex: [D, nan]) 
        
    Flag Overwrite
        flag_overwrite: Whether to overwrite the flag column as "D". (True or False.)

    Threshold
        threshold: Edit distance threshold for judging whether apply vectoring or not (0 to 100). if the score is less than the threshold, the vectoring result is applied. (Ex: 90) 
    
    Mode
        eval: Evaluation mode calculating the accuracies if True. (True or False.)
        eval_data_count: the number of data for evaluation. (Ex: 10000)
        inference: Inference mode that outputs the results if True. (True or False.)
"""

"""Input Example
python main.py --cfg configs/your_config.yaml
rye run python src/main.py --cfg configs/your_config.yaml
rye run python src/main.py --cfg configs/config_templates.yaml
"""

if __name__ == "__main__":
    # 現在のファイルが置かれているディレクトリのパスを取得
    current_dir = os.path.dirname(os.path.abspath(__file__))

    
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument("--cfg", type=str, required=True, help="Path to the configuration file")
    
    args = parser.parse_args()
    
    with open(args.cfg, "r") as f:
        cfg = yaml.safe_load(f)
        
    cfg = SimpleNamespace(**cfg)
        
    # コマンドライン引数から値のリストを取得
    cfg.source_flag = [str(i) if i.lower() != "nan" else np.nan 
                        for i in cfg.source_flag.strip("[]").split(",")]
    cfg.target_flag = [str(i) if i.lower() != "nan" else np.nan 
                        for i in cfg.target_flag.strip("[]").split(",")]

    print(f"-Input Information----------------")
    print(f"Target Column: {cfg.target}")
    print(f"Flag Column: {cfg.flag_col}")    
    print(f"Source Flag: {cfg.source_flag}")
    print(f"Target Flag: {cfg.target_flag}")
    print("----------------------------------")
    
    print("data loading...")
    df_original = pd.read_excel(cfg.input, index_col=0)
    print("data loading done.")
    
    df = df_original[[cfg.id, cfg.source, cfg.target, cfg.flag_col]]
    print("-Target Data----------------------------------")
    print(df.head(5))
    print("----------------------------------------------")
    
    
    train_data = df[df[cfg.flag_col].isin(cfg.source_flag)]
    # train_data = train_data[train_data[cfg.target]!="-1"]
    # train_data = train_data[train_data[cfg.target]!=-1]
    # train_data = train_data[train_data[cfg.target]!="[ERR]"]
    
    ####################################################################################
    ###### Evaluation
    ####################################################################################
    if cfg.eval:
        print("######################################################")
        print("Evaluation Mode")
        print("######################################################")
        df_sample = train_data[df[cfg.flag_col].isin(cfg.source_flag)]
        
        if len(train_data) > cfg.eval_data_count:
            df_sample = df_sample.sample(cfg.eval_data_count, random_state=42)
                
        sample_train_data, sample_test_data = train_test_split(df_sample, test_size=0.2, random_state=42)
            
        sample_train_data[[cfg.id, cfg.source, cfg.target, cfg.flag_col]].to_csv("train_data_sample.csv")

        col_id_words = list(sample_test_data.index)
        id_words = list(sample_test_data[cfg.id])
        words = [str(i) for i in sample_test_data[cfg.source].tolist()]
        gold_normalized = sample_test_data[cfg.target].tolist()
        
        ## Normalize entities
        print("normalizing...")    
        normalization_dictionary  = EntityDictionary(\
        'train_data_sample.csv',  cfg.source,  cfg.target)
        
        normalized, scores  = normalize(words,  normalization_dictionary,  matching_threshold=0)
        print("normalizing done.")
        
        normalization_dictionary  = VectorEntityDictionary(\
        'train_data_sample.csv',  cfg.source,  cfg.target, 'alabnii/jmedroberta-base-sentencepiece', f'{current_dir}/model/model.pth')
        
        vector_normalized, vector_scores  = vector_normalize(words,  normalization_dictionary)

        df_results_sample = pd.DataFrame([col_id_words, id_words, words, gold_normalized, normalized, scores, vector_normalized, vector_scores]).T
        df_results_sample.columns = ["行ID", cfg.id, cfg.source, "gold_normalized", "edit_distance_normalized", "edit_distance_score", "vector_normalized", "vector_score"]
        df_results_sample.set_index("行ID", inplace=True)
    
        
        added_EV_normalized = []
        for i, (norm, score, vector_norm, vector_score) in enumerate(zip(normalized, scores, vector_normalized, vector_scores)):
            if score < cfg.threshold:
                # print(f"{cfg.threshold}未満")
                # print(vector_norm, score)
                added_EV_normalized.append(vector_norm)
            else:
                # print(f"{cfg.threshold}以上")
                # print(norm, score)
                added_EV_normalized.append(norm)
        df_results_sample['EV_normalized'] = added_EV_normalized
        
        df_results_sample.to_csv("evaluation_inference_with_score.csv", index=False)
        
        # accuracyを計算
        accuracy_EV = accuracy_score([str(i) for i in df_results_sample['gold_normalized']], [str(i) for i in df_results_sample['EV_normalized']])
        accuracy_E = accuracy_score([str(i) for i in df_results_sample['gold_normalized']], [str(i) for i in df_results_sample['edit_distance_normalized']])
        accuracy_V = accuracy_score([str(i) for i in df_results_sample['gold_normalized']], [str(i) for i  in df_results_sample['vector_normalized']])
        
        # 結果をDataFrameにまとめる
        results = pd.DataFrame({
            'Metric': ['accuracy_EV', 'accuracy_E', 'accuracy_V'],
            f'TH={cfg.threshold}/LEN={len(df_sample)}': [accuracy_EV, accuracy_E, accuracy_V]
        })
        # 結果をCSVファイルに出力
        results.to_csv('evaluation_metrics.csv', index=False)
        
        # ファイルが存在するか確認して削除
        if os.path.exists('train_data_sample.csv'):
            os.remove('train_data_sample.csv')
            
        
        print("Evaluation")
        print(results)
        print("Evaluation done.")
    
    ####################################################################################
    ###### Full Data
    ####################################################################################
    if cfg.inference:
        print("######################################################")
        print("Inference Mode")
        print("######################################################")
            
        train_data[[cfg.id, cfg.source, cfg.target, cfg.flag_col]].to_csv("train_data.csv")
        
        col_id_words = list(df[df[cfg.flag_col].isin(cfg.target_flag)].index)
        id_words = list(df[df[cfg.flag_col].isin(cfg.target_flag)][cfg.id])
        words = df[df[cfg.flag_col].isin(cfg.target_flag)][cfg.source].tolist()
        words = [str(i) for i in words]

        ## Normalize entities
        print("normalizing...")    
        normalization_dictionary  = EntityDictionary(\
        'train_data.csv',  cfg.source,  cfg.target)
        
        normalized, scores  = normalize(words,  normalization_dictionary,  matching_threshold=0)
        print("normalizing done.")
        
        normalization_dictionary  = VectorEntityDictionary(\
        'train_data.csv',  cfg.source,  cfg.target, 'alabnii/jmedroberta-base-sentencepiece', f'{current_dir}/model/model.pth')
        
        vector_normalized, vector_scores  = vector_normalize(words,  normalization_dictionary)

        
        # ファイルが存在するか確認して削除
        if os.path.exists('train_data.csv'):
            os.remove('train_data.csv')


        df_results = pd.DataFrame([col_id_words, id_words, words, normalized, scores, vector_normalized, vector_scores]).T
        df_results.columns = ["行ID", cfg.id, cfg.source, "edit_distance_normalized", "score", "vector_normalized", "vector_score"]
        df_results.set_index("行ID", inplace=True)
        
        df_results.to_csv(cfg.output.split('.')[0] + "_inference_with_score.csv", index=False)
        
        idx = df_results.index
        for i, (idx, score, vector_normalized) in enumerate(zip(idx, scores, vector_normalized)):
            if score < cfg.threshold:
                df_results.loc[idx, "edit_distance_normalized"] = vector_normalized
        df_results.rename(columns={"edit_distance_normalized": cfg.target}, inplace=True)
        
        if cfg.flag_overwrite:
            print("Flag Overwrite")
            df_results[cfg.flag_col] = len(df_results)*["D"]
        # df_results.to_csv("inference.csv")
        
        print("-Modified Data--------------------------------")
        print(df_results.head(5))
        print("----------------------------------------------")
        
        # DataFrameをアップデートする前のコピーを作成
        df_original_copy = df_original.copy()

        # マージ
        df_original.update(df_results)
        
        # 実際に変更された要素の数を計算する
        matching_elements = [(i, z, x, y) for i, (z, x, y) in \
            enumerate(zip(df_original[cfg.source], df_original_copy[cfg.target], df_original[cfg.target])) \
            if x != y and not (pd.isnull(x) and pd.isnull(y))]

        print("更新された要素(最初の10件)(行番号，出現形，更新前用語，更新後用語)")
        pprint.pprint(matching_elements[:10])
        print("更新された要素の数:", len(matching_elements))
        print(f"更新された要素の割合:{len(matching_elements)/len(df_original)*100:.1f}%")
        # pd.DataFrame(matching_elements).to_csv("matching_elements.csv")

        ## Write output file
        print("saving...")
        df_original.to_excel(cfg.output)
        print("saving done.")