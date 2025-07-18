from utils.data import NERDataLoader, NERDataset

if __name__ == '__main__':
    # Label list without website for testing old model
    LABEL_LIST = ["O", "B-Name", "I-Name", "B-Position", "I-Position", "B-Company", 
                "I-Company", "B-Address", "I-Address", "B-Phone", "I-Phone", "B-Email", 
                "I-Email", "B-Department", "I-Department"]