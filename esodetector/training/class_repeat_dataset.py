from typing import Optional

from timm.data import ImageDataset


class ClassRepeatDataset(ImageDataset):
    """Wrapper dataset that repeats samples from specific classes.
    
    This is useful for handling class imbalance by oversampling minority classes.
    """

    def __init__(self, class_to_repeat: Optional[list[int]] = None, repeat_factor=1, **kwargs):
        """
        Args:
            class_to_repeat: Class index to repeat (list of ints). If None, no repetition.
            repeat_factor: How many times to repeat samples from the specified class(es) (default: 1, no repetition)
        """
        super().__init__(**kwargs)
        self.repeat_factor = max(1, repeat_factor)
        self.class_to_repeat = [] if class_to_repeat is None else class_to_repeat
        
        # Build index mapping
        self.indices = []
        original_length = super().__len__()
        
        for idx in range(original_length):
            # Get the label from the dataset
            _, label = self.reader.samples[idx]
            
            if label in self.class_to_repeat:
                # Repeat this sample multiple times
                self.indices.extend([idx] * self.repeat_factor)
            else:
                self.indices.append(idx)
        
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        original_idx = self.indices[idx]
        return super().__getitem__(original_idx)