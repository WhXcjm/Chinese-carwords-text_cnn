import torch
import torch.nn.functional as F
import pickle
import model as MD
import dataset
import argparse
import torchtext.data as data
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

class Predictor:
	def __init__(self, model_path='saved/best_model.pt', args_path='saved/args.pkl', vocab_path='saved/text_field_vocab.pkl', device=None):
		print('Loading model...')
		self.device = device
		# Load args
		with open(args_path, 'rb') as f:
			self.args = pickle.load(f)
		if(self.device):
			self.args.device = self.device
		# Load vocab
		with open(vocab_path, 'rb') as f:
			text_field_vocab = pickle.load(f)
		self.text_field = data.Field(lower=True, tokenize=dataset.word_cut)  # 使用分词函数
		self.text_field.vocab = text_field_vocab
		if self.args.static:
			self.args.embedding_dim = self.text_field.vocab.vectors.size()[-1]
			self.args.vectors = self.text_field.vocab.vectors
		# Load model
		self.model = MD.TextCNN(self.args)
		self.model.load_state_dict(torch.load(model_path))
		if self.args.cuda:
			torch.cuda.set_device(self.args.device)
			self.model = self.model.cuda()
		print('Model loaded')

	def predictstr(self, senlist):
		self.model.eval()
		processed_texts = [self.text_field.preprocess(text) for text in senlist]
		tokenized_texts = self.text_field.process(processed_texts).to(self.args.device).t_()

		predictions = []
		with torch.no_grad():
			logits = self.model(tokenized_texts)
			probabilities = F.softmax(logits, dim=-1)
			predicted_labels = torch.max(probabilities, 1)[1]
			predictions.extend(predicted_labels.cpu().numpy())

		return predictions


def main():
	parser = argparse.ArgumentParser(description='TextCNN text classifier')
	parser.add_argument('-model-path', type=str, default='saved/best_model.pt', help='Path to the best model')
	parser.add_argument('-args-path', type=str, default='saved/args.pkl', help='Path to the args file')
	parser.add_argument('-vocab-path', type=str, default='saved/text_field_vocab.pkl', help='Path to the vocab file')
	parser.add_argument('-device', type=str, default=None, help='Device to use for prediction, e.g., "cpu" or "cuda:0"')
	args = parser.parse_args()

	predictor = Predictor(model_path=args.model_path, args_path=args.args_path, vocab_path=args.vocab_path, device=args.device)
	
	# Sample sentences to predict
	test_sentences = [
    "挺时尚风格的一款车子，特别的大气，开出去挺有面子的，而且这款车子的运动风格是比较足的。",  # 正面
    "隔音效果做的并不是很到位，而且这款车子的车机方面稍微有点偏薄了，并不是很厚。",  # 负面
    "空间方面足够用了，这款车子的前后排空间都是设计非常合理的，而且后备箱的空间也是比较大的。",  # 正面
    "操控性能挺好的，提速快，平顺性很好。",  # 正面
    "是比较有性价比的车子，外观方面非常的给力，这款车子的设计符合年轻人的眼光，特别的大气。",  # 正面
    "对于这款车子不满意的，就是速度上来的时候能够听到明显的胎噪声音在隔音这一块并不是很好。",  # 负面
    "座椅方面还是挺柔软的，而且腰部的支撑性非常的强，这款车子它的平稳性也是比较不错的。",  # 正面
    "油耗方面也是比较低的，这款车子它在高速公路上的油耗是不到6个，还是挺划算的。",  # 正面
    "感觉这款车子还是挺不错的，尤其是在起步的时候，速度还是比较快的，在高速公路上经常开也是能够感觉出动力方面的充足。",  # 正面
    "配置方面还是可以的，它的驾驶模式这一块是可以切换的，并且一些常用的配置都很齐全。"  # 正面
    "这款车的配置中缺少360度全景影像和中控屏幕尺寸较小。作为车主，我希望能够更好地了解周围环境，而全景影像能够提供更全面的视野。同时，更大尺寸的中控屏幕会提升操作体验。",  # 负面
	"馈电状态下启动发动机会有很大的轰鸣声，感觉比较突兀。",
	"但是车漆有些薄，平常开下来很容易划伤，要特别注意。"
]

	predictions = predictor.predictstr(test_sentences)
	for prediction, sentence in zip(predictions, test_sentences):
		label = "正面" if prediction == 1 else "负面"
		print(f"{label}: {sentence}")

if __name__ == "__main__":
	main()
