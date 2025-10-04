# 这些列表用于定义各数据集的保持类别(Preserved Classes)
# 在遗忘学习中，当遗忘指定类别时，需要确保其他类别的性能不受影响

# 当前项目中暂未使用的数据集类别列表 - 可以根据需要启用
caltech_list = [] 
stanfordcars_list = []
oxfordflowers_list = []
stanforddogs_list = []

# 如果需要使用其他数据集，可以取消注释并添加相应的类别列表


# ============== 当前项目使用的数据集类别列表 ==============

# PLTNetMini植物数据集的所有类别 (25个植物类)
pltnetmini_list = [
    "Aegopodium_podagraria",
    "Alcea_rosea",
    "Alliaria_petiolata",
    "Anemone_nemorosa",
    "Calendula_officinalis",
    "Centranthus_ruber",
    "Cirsium_arvense",
    "Cirsium_vulgare",
    "Cymbalaria_muralis",
    "Daucus_carota",
    "Fragaria_vesca",
    "Hypericum_perforatum",
    "Lactuca_serriola",
    "Lamium_galeobdolon",
    "Lamium_purpureum",
    "Lapsana_communis",
    "Lavandula_angustifolia",
    "Papaver_rhoeas",
    "Papaver_somniferum",
    "Punica_granatum",
    "Pyracantha_coccinea",
    "Sedum_album",
    "Tagetes_erecta",
    "Trifolium_pratense",
    "Trifolium_repens"
]

# Bird525数据集使用数字ID作为类别标识，这里不需要预定义类别列表
# Bird525的保持类别将在运行时从数据集动态获取
# 注意: forget_cls.py 中定义了要遗忘的鸟类ID: ['0', '10', '42']

# Bird525类别列表 - 占位符，将由动态获取替代
# 这里提供一个空列表，实际使用时会从dataset.classnames动态获取
bird525_list = []  # 525个鸟类，运行时动态填充

# ============== 动态类别获取函数 ==============

def get_class_lists(datasets_cls):
    """
    获取所有数据集的类别列表，包括动态获取的Bird525类别
    
    Args:
        datasets_cls: 数据集类的字典，包含classnames属性
    
    Returns:
        dict: 包含所有数据集类别列表的字典
    """
    class_lists = {
        'PLTNetMini': pltnetmini_list,
        'StanfordDogs': stanforddogs_list, 
        'StanfordCars': stanfordcars_list,
        'Caltech101': caltech_list,
        'OxfordFlowers': oxfordflowers_list
    }
    
    # 动态获取Bird525的类别列表
    if 'Bird525' in datasets_cls:
        class_lists['Bird525'] = datasets_cls['Bird525'].classnames
    
    return class_lists

# ============== 功能说明 ==============
# 这些列表的作用:
# 1. 在遗忘学习过程中定义需要保持性能的类别
# 2. 构造preserve_loss，确保遗忘指定类别时其他类别不受影响  
# 3. get_preserved_classes()函数会排除遗忘类别，返回需要保持的类别列表