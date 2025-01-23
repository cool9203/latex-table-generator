[
    {"names": ["編號", "#"], "type": "int", "empty": False, "hashtag": False, "sequence": True, "range": None, "choices": None},
    {
        "names": ["部位"],
        "type": "str",
        "empty": False,
        "hashtag": False,
        "sequence": False,
        "range": None,
        "choices": [
            "樑",
            "柱",
            "版",
            "牆",
            "梯",
            "上版",
            "下版",
            "樑料",
            "柱料",
            "版料",
            "牆料",
            "梯料",
            "樑補",
            "柱補",
            "版補",
            "牆補",
            "梯補",
            "上版補",
            "下版補",
            "梯1",
            "梯2",
            "梯3",
            "電梯牆",
            "下層版筋",
        ],
    },
    {"names": ["號數"], "type": "int", "empty": False, "hashtag": True, "sequence": False, "range": (1, 20), "choices": None},
    {
        "names": [
            "圖示",
            "施工內容",
            "加工形狀",
            "加工型狀",
            "加工形式",
            "加工型式",
            "形狀",
            "型狀",
            "形式",
            "型式",
            "料型",
            "料形",
        ],
        "type": "str",
        "empty": False,
        "hashtag": False,
        "sequence": False,
        "range": None,
        "choices": None,
    },
    {
        "names": ["長度", "長度(cm)", "總長度", "料長"],
        "type": "int",
        "empty": False,
        "hashtag": False,
        "sequence": False,
        "range": (1, 2000),
        "choices": None,
    },
    {
        "names": ["數量", "支數"],
        "type": "int",
        "empty": False,
        "hashtag": False,
        "sequence": False,
        "range": (1, 2000),
        "choices": None,
    },
    {
        "names": ["重量", "重量(kg)", "重量Kg", "重量噸"],
        "type": "int",
        "empty": False,
        "hashtag": False,
        "sequence": False,
        "range": (1, 20000),
        "choices": None,
    },
    {"names": ["備註"], "type": "str", "empty": True, "hashtag": False, "sequence": False, "range": None, "choices": None},
]
