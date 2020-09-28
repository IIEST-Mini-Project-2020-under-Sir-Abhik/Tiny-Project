import numpy as np
import sys
from PIL import Image

a = []
size = (50, 50)
image_name = input("Enter the name/path of the image: ")
images = np.load('processed_data.npy',allow_pickle=True)    #loading processed data

#testing if the loaded data is correct or not
def test_loaded_data(): #run this while debugging
    for x in list(images):
        np.savetxt(sys.stdout, x, fmt='%i')
        print()

def comp(i):
    s = [ images[i][x][y]==a[x][y] for x in range(50) for y in range(50) if a[x][y] ]
    return sum(s)/len(s)

def main():
    global a
    with Image.open(image_name) as img:
        img = img.resize(size).convert('L') #resizing it according to our convenience
    a = np.array((np.asarray(img))<129, dtype=int)

    # np.savetxt(sys.stdout, a, fmt='%i')

    res = np.array([comp(i) for i in range(10)])
    print("Prediction: ", np.argmax(res))

main()