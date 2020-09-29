try:    #if numpy is not found
    import numpy as np
except Exception as e:
    print(e)
    print("Install numpy with 'pip3 install numpy' command (or pip)"); exit()

try:    #if PIL is not found
    from PIL import Image
except Exception as e:
    print(e)
    print("Install PIL with 'pip3 install pillow' command (or pip)"); exit()

size = (50, 50)
images = np.load('processed_data.npy',allow_pickle=True)    #loading processed data

def comp(i):
    s = [ images[i][x][y]==a[x][y] for x in range(50) for y in range(50) if a[x][y] ]
    return sum(s)/len(s)

s = []
for i in range(10):
    with Image.open(f'test_data/{i}.jpg') as img:
        img = img.resize(size).convert('L') #resizing it according to our convenience
        a = np.array((np.asarray(img))<129, dtype=int)
    res = np.array([comp(j) for j in range(10)])
    s.append((i==np.argmax(res)))

# print(s)
print("Accuracy: ", sum(s)*10,'%', sep = '') #60%
print("The incorrect numbers are:", [i for i in range(10) if not s[i]])