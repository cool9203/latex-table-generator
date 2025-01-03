SEED=99 LOG_LEVEL=INFO \
    uv run src/latex_table_generator \
    --input_path \
        /mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241213_需Label鋼材Data/單一正規 \
        /mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241216_需Label鋼材Data/1_單一正規 \
        /mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241218_需Label鋼材Data/單一正規 \
        /mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241223_需Label鋼材Data/1_單正規表格 \
        /mnt/c/Users/ychsu/Downloads/沛波標記data/要標記資料/20241225_需Label鋼材Data/1_單正規表格 \
    --output_path ./outputs \
    --v_contents "工作筋" "鋼材筋" "寬止筋" "含上層工作筋" "增築" "上增築" "穿樑補強" "角隅增強" "角隅增強*120" \
        "已扣約100支至x向" "CS花台" "柱擴柱" "門窗補強" "電梯剪力牆" "車道" "車道預留*10" "備料5支" "備料7支" "備料10支" \
        "另外綑綁" "甘蔗筋" \
    --h_contents "開口補強" "請依區別分開包裝" "請依施工內容分開包裝" "分開綑綁" "雜項鋼筋" "增築" \
    --vh_contents "工作筋" "鋼材筋" "寬止筋" "含上層工作筋" "增築" "上增築" "穿樑補強" "角隅增強" "角隅增強*120" \
        "CS花台" "柱擴柱" "門窗補強" "電梯剪力牆" "車道" "車道預留*10" "甘蔗筋" "開口補強" "增築" \
    --image_path "./steels" \
    --horizontal_count 1 3 \
    --vertical_count 1 3 \
    --tqdm
