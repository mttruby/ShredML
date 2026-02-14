from ml.train.train_lstm import train_lstm_cv
from ml.nodes.preprocessor.yolo_preprocessor import YOLOPreprocessor

def preprocess():
    vp = YOLOPreprocessor(yolo_model="yolo26n.pt", use_gpu=True)
    vp.preprocess_videos("../data/Kickflip")
    vp.preprocess_videos("../data/Ollie")

if __name__ == "__main__":

    # preprocess()
    train_lstm_cv()
    #train_3dcnn()