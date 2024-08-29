# EDV-ExampleBasedCoding


## Config arguments:

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
        threshold: Edit distance threshold for judging whether apply vectoring or not (0 to 100). if the score is less than the threshold, the vectorizing result is applied. (Ex: 90) 
    
    Mode
        eval: Evaluation mode calculating the accuracies if True. (True or False.)
        eval_data_count: the number of data for evaluation. (Ex: 10000)
        inference: Inference mode that outputs the results if True. (True or False.)

## Command example with arguments:

    python main.py --cfg configs/your_config.yaml
    rye run python src/main.py --cfg configs/your_config.yaml
    rye run python src/main.py --cfg configs/config_templates.yaml



## How to use
1. レポジトリのクローン
    ```
    git clone https://github.com/sociocom/EDV-ExampleBasedCoding.git
    ```

    ~~model.pthファイルがクローン後に正しく取得されていない場合，クローン後に以下のコマンドで手動で取得する．~~
   制限に達すると，ダウンロードできなくなるのでLFSは使わず，手動でダウンロードする．
    ```
    git lfs pull
    ```

   ~~lfsについては，データ容量の制限に達している場合，以下のようなエラーが出る．~~
   ```
   batch response: This repository is over its data quota. Account responsible for LFS bandwidth should purchase more data packs to restore access.                                
   error: failed to fetch some objects from 'https://github.com/sociocom/EDV-ExampleBasedCoding.git/info/lfs'
   ```

   Huggingfaceのサイトから手動でダウンロードする．
   
   https://huggingface.co/sociocom/EDV-ExampleBasedCoding
    
3. Rye を導入した後，必要なパッケージのインストール
    ```
    rye sync
    ```
4. 仮想環境を立ち上げる．
    ```
    source .venv/bin/activate
    ```
5. config_template.yamlをコピーし設定ファイルを作成（設定を必要に応じて変更してください）
    ```
    cp configs/config_templates.yaml configs/your_config.yaml
    ```
6. スクリプトの実行
    ```
    rye run python src/main.py --cfg configs/your_config.yaml
    ```
