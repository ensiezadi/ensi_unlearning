import os
import pandas as pd
from collections import defaultdict

from dassl.data.datasets import DATASET_REGISTRY, Datum, DatasetBase
from dassl.utils import mkdir_if_missing

@DATASET_REGISTRY.register()
class Bird525(DatasetBase):
    """
    Bird525 dataset.
    
    This loader is specifically adapted for datasets that are already split
    into train, validation, and test directories. It uses a master CSV file
    to map image paths to their respective labels and splits.
    """
    
    # 指定你的数据集文件夹名称
    dataset_dir = "bird_525"

    def __init__(self, cfg):
        root = os.path.abspath(os.path.expanduser(cfg.DATASET.ROOT))
        self.dataset_dir = os.path.join(root, self.dataset_dir)
        
        # 定义关键文件的路径
        self.csv_path = os.path.join(self.dataset_dir, "birds.csv")
        self.image_dir = self.dataset_dir # 图片的根目录就是数据集目录

        if not os.path.exists(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found at: {self.dataset_dir}")
        if not os.path.exists(self.csv_path):
             raise FileNotFoundError(f"Master CSV file not found at: {self.csv_path}")

        train, val, test = self._read_data()
        
        super().__init__(train_x=train, val=val, test=test)

    def _read_data(self):
        """
        Reads data from pre-defined splits (train, valid, test) based on birds.csv.
        """
        try:
            df = pd.read_csv(self.csv_path)
            print(f"CSV file columns: {df.columns.tolist()}")
            print(f"CSV file shape: {df.shape}")
            
            # 检查必需的列是否存在并推断列名
            label_col = None
            filepath_col = None
            split_col = None
            
            for col in df.columns:
                col_lower = col.lower()
                if 'label' in col_lower or 'class' in col_lower:
                    label_col = col
                elif 'path' in col_lower or 'file' in col_lower:
                    filepath_col = col
                elif 'split' in col_lower or 'set' in col_lower:
                    split_col = col
            
            if not label_col or not filepath_col:
                raise ValueError(f"Could not find required columns. Available: {df.columns.tolist()}")
            
            print(f"Using columns: label='{label_col}', filepath='{filepath_col}', split='{split_col}'")
            
            # 如果标签是数字ID，需要创建类别名称映射
            unique_labels = sorted(df[label_col].unique())
            if all(isinstance(label, (int, float)) for label in unique_labels):
                # 数字标签，创建类别名称
                print("Detected numeric labels, creating class names...")
                # 从目录结构推断类别名称
                try:
                    train_dir = os.path.join(self.dataset_dir, 'train')
                    if os.path.exists(train_dir):
                        # 假设train目录下每个子目录对应一个类别
                        class_dirs = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
                        class_dirs = sorted(class_dirs, key=int)  # 按数字顺序排序
                        
                        # 创建ID到类别名的映射（使用实际的class_id作为键）
                        id_to_class = {int(class_dir): f"bird_class_{class_dir}" for class_dir in class_dirs}
                        df['class_name'] = df[label_col].map(id_to_class)
                        label_col = 'class_name'
                        print(f"Created class mapping for {len(id_to_class)} classes")
                    else:
                        # 如果没有train目录，使用通用命名
                        id_to_class = {label: f"bird_class_{int(label)}" for label in unique_labels}
                        df['class_name'] = df[label_col].map(id_to_class)
                        label_col = 'class_name'
                        print(f"Created generic class names for {len(id_to_class)} classes")
                except Exception as e:
                    print(f"Error creating class names: {e}")
                    # 最后的备选方案
                    id_to_class = {label: f"class_{int(label)}" for label in unique_labels}
                    df['class_name'] = df[label_col].map(id_to_class)
                    label_col = 'class_name'
        
            # 从 labels 列获取所有唯一的类别名称并排序
            class_names = sorted(df[label_col].unique())
            # 确保所有类别名称都是字符串
            class_names = [str(name) for name in class_names]
            self._classnames = class_names
            class_to_idx = {name: i for i, name in enumerate(class_names)}
            
            print(f"Sample class names: {class_names[:5]}...")
            print(f"All class names are strings: {all(isinstance(name, str) for name in class_names)}")
            
        except Exception as e:
            print(f"Error reading CSV file {self.csv_path}: {e}")
            raise
        
        print(f"Found {len(class_names)} classes.")

        train_data = []
        val_data = []
        test_data = []

        # 创建一个辅助函数来处理每个数据分割
        def populate_split(split_name, target_list):
            # 从DataFrame中筛选出属于当前split的数据
            if split_col:
                split_df = df[df[split_col] == split_name]
            else:
                # 如果没有split列，假设所有数据都是train
                if split_name == 'train':
                    split_df = df
                else:
                    split_df = df.iloc[0:0]  # 空DataFrame
            
            print(f"Processing {split_name} split with {len(split_df)} images...")
            
            for _, row in split_df.iterrows():
                # 从CSV中获取相对路径和类别名称
                relative_impath = row[filepath_col]
                classname = str(row[label_col])  # 确保类别名称是字符串
                
                # 检查classname是否在class_to_idx中
                if classname not in class_to_idx:
                    print(f"Warning: Class '{classname}' not found in class_to_idx mapping")
                    continue
                
                # 构建完整的图像绝对路径
                full_impath = os.path.join(self.image_dir, relative_impath)
                
                # 如果路径不存在，尝试修复常见的拼写错误
                if not os.path.exists(full_impath):
                    # 尝试修复已知的拼写错误
                    corrected_path = relative_impath
                    
                    # 修复PARAKETT  AKULET -> PARAKETT  AUKLET (双空格+AKULET -> 双空格+AUKLET)  
                    corrected_path = corrected_path.replace('PARAKETT  AKULET', 'PARAKETT  AUKLET')
                    
                    # 特殊修复：valid分割中PARAKETT  AUKLET (双空格) -> PARAKETT AUKLET (单空格)
                    if 'valid/' in corrected_path:
                        corrected_path = corrected_path.replace('PARAKETT  AUKLET', 'PARAKETT AUKLET')
                    
                    # 可以在这里添加更多的拼写错误修复规则
                    # corrected_path = corrected_path.replace('OTHER_TYPO', 'CORRECT_NAME')
                    
                    if corrected_path != relative_impath:
                        full_impath = os.path.join(self.image_dir, corrected_path)
                        if os.path.exists(full_impath):
                            # print(f"✅ Path corrected: {relative_impath} -> {corrected_path}")
                            continue
                        else:
                            print(f"Warning: Image not found even after path correction: {full_impath}")
                            continue
                    else:
                        print(f"Warning: Image not found: {full_impath}")
                        continue
                
                item = Datum(
                    impath=full_impath,
                    label=class_to_idx[classname],
                    classname=classname
                )
                target_list.append(item)

        # 分别填充 train, valid, 和 test 列表
        populate_split('train', train_data)
        populate_split('valid', val_data)
        
        # 如果没有test split，使用valid作为test
        available_splits = df[split_col].unique() if split_col else []
        if 'test' in available_splits:
            populate_split('test', test_data)
        else:
            print("No 'test' split found, using 'valid' split as test data")
            populate_split('valid', test_data)
        
        print(f"Dataset loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test images.")
        
        return train_data, val_data, test_data
    
    @property
    def classnames(self):
        return self._classnames