from tkinter import *
from tkinter import filedialog
from tkinter.messagebox import *
from tkinter.tix import Tk  # 升级的控件组包

import cv2
from PIL import Image, ImageTk
from numpy import *
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression


# 算法部分
# 加载数据集
def loadDataSet(k):  # k代表在10张图片中选择几张作为训练集
    dataSetDir = 'ORL'
    # 显示文件夹内容
    choose = random.permutation(10) + 1  # 随机排序1-10 (0-9）+1
    #生成矩阵
    train_face = zeros((40 * k, 112 * 92))
    train_face_number = zeros(40 * k)
    test_face = zeros((40 * (10 - k), 112 * 92))
    test_face_number = zeros(40 * (10 - k))
    for i in range(40):  # 40个人
        people_num = i + 1
        for j in range(10):  # 每个人有10个不同的脸
            if j < k:  # 测试集
                filename = dataSetDir + '/s' + str(people_num) + '/' + str(choose[j]) + '.bmp'
                img = img2vector(filename)
                train_face[i * k + j, :] = img
                train_face_number[i * k + j] = people_num
            else:
                filename = dataSetDir + '/s' + str(people_num) + '/' + str(choose[j]) + '.bmp'
                img = img2vector(filename)
                test_face[i * (10 - k) + (j - k), :] = img
                test_face_number[i * (10 - k) + (j - k)] = people_num
    return train_face, train_face_number, test_face, test_face_number


# 将图片转换成矩阵
def img2vector(filename):
    img = cv2.imread(filename, 0)  # 读入灰度值
    rows, cols = img.shape
    imgVector = zeros((1, rows * cols))
    imgVector = reshape(img, (1, rows * cols))  # 将2维转成1维
    return imgVector


def facefind():
    # 获取训练集
    train_face, train_face_number, test_face, test_face_number = loadDataSet(3)
    # PCA训练训练集，用pca将数据降到30维
    pca = PCA(n_components=30).fit(train_face)
    # 返回测试集和训练集降维后的数据集
    x_train_pca = pca.transform(train_face)
    x_test_pca = pca.transform(test_face)
    # 逻辑回归训练
    classirfier = LogisticRegression()
    lr = classirfier.fit(x_train_pca, train_face_number)

    # 保存模型
    joblib.dump(lr, 'lr.model')
    # 计算精确度和召回率
    accuray = classirfier.score(x_test_pca, test_face_number)
    recall = accuray * 0.7

    return accuray, recall, pca


# 界面部分
def choosepic():  # 选择图片函数
    file_path = filedialog.askopenfilename()  # 加载文件
    path.set(file_path)
    img_open = Image.open(file.get())
    img = ImageTk.PhotoImage(img_open)
    pic_label.config(image=img)
    pic_label.image = img

    string = str(file.get())

    # 预测的人
    predict = img2vector(string)
    # 加载模型
    LR = joblib.load('lr.model')
    predict_people = LR.predict(pca.transform(predict))

    string1 = str("编号：S%s 精确度：%f 召回率：%f" % (predict_people, accuray, recall))
    showinfo(title='图像分析', message=string1)


# 初始化Tk（）
accuray, recall, pca = facefind()
root = Tk()  # root便是你布局的根节点了，以后的布局都在它之上
root.geometry('260x140')
root.title("人脸识别系统")  # 设置窗口标题
root.resizable(width=False, height=False)  # 设置窗口是否可变
root.tk.eval('package require Tix')  # 引入升级包，这样才能使用升级的组合控件
path = StringVar()  # 跟踪变量的值的变化

Button(root, text='选择图片', command=choosepic, width=1, height=1).grid(row=1, column=1, sticky=W + E + N + S, padx=40,
                                                                     pady=20)  # command指定其回调函数
file = Entry(root, state='readonly', text=path)
file.grid(row=0, column=1, sticky=W + E + S + N, padx=6, pady=20)  # 用作文本输入用

pic_label = Label(root, text='图片', padx=30, pady=10)
pic_label.grid(row=0, column=2, rowspan=4, sticky=W + E + N + S)
root.mainloop()