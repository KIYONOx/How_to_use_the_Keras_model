import sys
import os
import shutil
import sys,datetime
from PIL import Image 
import numpy as np
from keras.models import model_from_json

def main(files):
	
	# model load
	model = model_from_json(open('mnist_mlp_model.json').read())
	# load weight
	model.load_weights('mnist_mlp_weights.h5')
	# image load
	img = Image.open(files)
	# img resize
	# imag change array
	resize_img = img.resize((28, 28))
	resize_img_array = np.array(resize_img.convert('L'), 'f' )
	ans_array = resize_img_array.reshape(1, 784).astype('float32') *255
	# model predict
	ans = model.predict(ans_array)
	# one-hot change int number
	y = - 1
	for x in ans[0]:
		y = y + 1
		if x == 1 :
			no = y
	print (no)
	return (no)


if __name__ == '__main__':
    files = sys.argv[1]
    main(files)