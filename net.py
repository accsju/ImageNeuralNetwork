import numpy as np
import math
import os
import random
from PIL import Image

## ニューラルネットワーク構成例
file_name = "3層3タスクニューラルネットワーク.npz" #重み、バイアスを保存するfile名
resize = {
    #入力サイズ
    "width": 100,
    "height": 100
}
network_input = 3 * resize["width"] * resize["height"] #　カラー画像を想定　3チャネル * 100(width) * 100(height)
train_set_size = 1000 #　テスト・学習する画像枚数
folders = [
    #学習データセットのフルパス
    #例　
    #"C:\\...\\grape",
    #"C:\\...\\banana",
    #"C:\\...\\apple"
]
folders_test = [
    #学習したモデルをテストするための画像データセットのフルパス
    #例　
    #"C:\\...\\test_grape",
    #"C:\\...\\test_banana",
    #"C:\\...\\test_apple"
]

task_classes = len(folders)
hyper_parm_composition = {
    "learning_rate": 0.001,#学習率
    "dropout": 0.5,#ドロップレート
}
net_composition = [ #ネットワーク構成
    {   
        "activation": "ReLU",
        "dropout": True
    },
    {   
        "activation": "ReLU",
        "dropout": True
    },
    {   
        "activation": "softmax",
        "dropout": False
    }
]
weight_recipe = [network_input,1000,500,task_classes] #ニューラルネットワークの重みレシピ
weight_composition = {#ReLu活性化関数を用いるので重みの初期値はHeを用いる
    "layer-length": len(weight_recipe)-1,
    "mean": 0.0,
    "weight-size-array": weight_recipe,
}
class train_image_set_up: #画像をセットするclass
    def __init__(self, train_set_size, folders, task_classes):
        self.train_set_size = train_set_size
        self.folders = folders 
        self.task_classes = task_classes
    def random_image_set_up(self): #フォルダから画像をランダムに取得、正解ラベルを作成して返す関数
        true_labels = []
        train_images = []
        for i in range(0,train_set_size):
            x = random.randint(0,task_classes-1)
            y = os.path.join(folders[x],os.listdir(folders[x])[random.randint(0,len(os.listdir(folders[x]))-1)])
            label = np.zeros(task_classes)
            label[x] = 1
            true_labels.append(label)
            train_images.append(y)
        return true_labels, train_images
    def image_flatten(self, train_images, resize): #画像を正則化、リサイズして適切な入力値を作成する関数
        arr = []
        for train_image in train_images:
            image = Image.open(train_image)
            image = image.resize((resize["width"], resize["height"]))
            image = np.array(image)
            image = image.flatten()/255
            arr.append(image)
        return arr
class activation_functions: #活性化関数と逆伝播活性化関数
    @classmethod
    def relu(self, x):
        return np.clip(x, 0, None)
    @classmethod
    def backward_relu(self, loss, activation_mask):
        loss[activation_mask] = 0
        return loss 
    @classmethod
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x)) 
        return exp_x / np.sum(exp_x)
    @classmethod
    def activation(self, activation_name,x):
        if activation_name == "ReLU":
            return activation_functions.relu(x)
        if activation_name == "softmax":
            return activation_functions.softmax(x)
        return print("no activation")
    @classmethod
    def backward_activation(self, activation_name, loss, activation_mask):
        if activation_name == "ReLU":
            return activation_functions.backward_relu(loss, activation_mask)
        if activation_name == "softmax":
            return loss
        return print("no activation")
class propagation(activation_functions): #
    def __init__(self, net_composition):
        self.net_composition = net_composition
    def forward_start(self, x, w, b, dropout):# 順伝播関数
        forward_x_cache = {}
        forward_activation_cache = {}
        forward_dropout_cache = {}
        transmission = x.reshape((1,len(x)))
        for i in range(0, len(net_composition)):
            forward_x_cache[f"x{i+1}"] = transmission
            transmission = np.dot(transmission, w[f"w{i+1}"]) + b[f"b{i+1}"]
            transmission = activation_functions.activation(net_composition[i]["activation"],transmission)
            if net_composition[i]["activation"] == "ReLU":
                forward_activation_cache[f"activation{i+1}"] = (transmission<=0)
            if net_composition[i]["activation"] == "softmax":
                forward_activation_cache[f"activation{i+1}"] = transmission
            if net_composition[i]["dropout"]:
                mask_dropout = np.random.rand(*transmission.shape)>dropout
                transmission*mask_dropout
                transmission*=(1-dropout)
                forward_dropout_cache[f"dropout{i+1}"] = mask_dropout
        return forward_x_cache, forward_activation_cache, forward_dropout_cache, transmission
    def aggregate(self, output, true_label): #正解集計関数
        solve_index = np.argmax(output)
        true_index = np.argmax(true_label)
        if solve_index == true_index:
            return 1
        return 0
    def evaluation_and_cost(self, output, true_label): #誤差関数
        delta = 1e-4
        print(f"交差エントロピー誤差:{np.sum(-true_label*np.log(output+delta))}",output)
        return output - true_label #逆伝播の起点となる値　修正誤差
    def backward_start(self, loss, forward_activation_cache, forward_dropout_cache, w): #逆伝播関数
        update_delta = {}
        for i in range(len(net_composition), 0, -1):
            if net_composition[i-1]["dropout"]:
                loss[forward_dropout_cache[f"dropout{i}"]] = 0
                update_delta[f"e{i}"] = loss 
            loss = activation_functions.backward_activation(net_composition[i-1]["activation"], loss, forward_activation_cache[f"activation{i}"])
            if not net_composition[i-1]["dropout"]:
                update_delta[f"e{i}"] = loss
            loss = np.dot(loss,w[f"w{i}"].T)
        return update_delta
    def weight_and_bias_update(self, w, b, learning_rate, forward_x_cache, update_delta): #重み、バイアス更新関数
        for i in range(0,len(update_delta)):
            w[f"w{i+1}"] -= learning_rate*np.dot(
                forward_x_cache[f"x{i+1}"].T,
                update_delta[f"e{i+1}"]
            ) 
            b[f"b{i+1}"] -= learning_rate*np.sum(
                update_delta[f"e{i+1}"],
                axis=0  
            )
    def weight_and_bias_save(self, file_name, w, b, layer_length):#重みとバイアス保存関数
        data_to_save = {}
        for i in range(0, len(w)):
            data_to_save.update({f'w{i+1}': w[f"w{i+1}"]})
            data_to_save.update({f'b{i+1}': b[f"b{i+1}"]})
        data_to_save.update({"layer-length": layer_length})
        np.savez(
            file_name,
            **data_to_save
        )    
class weight_bias_set_up:
    def __init__(self, weight_composition):
        self.weight_composition = weight_composition
    def weight_create(self, weight_composition):#重みデータがない場合重みを作成する関数
        weight = {}
        bias = {}
        for i in range(0,weight_composition["layer-length"]):
            weight[f"w{i+1}"] = np.random.normal(
                loc=weight_composition["mean"],
                scale=math.sqrt(2/weight_composition["weight-size-array"][i]),
                size=(
                weight_composition["weight-size-array"][i],
                weight_composition["weight-size-array"][i+1])
            )
            bias[f"b{i+1}"] = np.zeros((1,weight_composition["weight-size-array"][i+1]))
        return weight, bias
    def weight_load(self, file_name):#重みデータを読み込む関数
        data = np.load(file_name)
        layer_length = data['layer-length']
        weight = {}
        bias = {}
        for i in range(0,layer_length):
            weight[f"w{i+1}"] = data[f"w{i+1}"]
            bias[f"b{i+1}"] = data[f"b{i+1}"]
        return weight, bias
def nn_execute(file_name, hyper_parm_composition, net_composition, train_set_size, resize):  
    train = False
    test = True
    if train:
        tisu = train_image_set_up(train_set_size, folders, task_classes)
    if test:
        tisu = train_image_set_up(train_set_size, folders_test, task_classes)
    true_labels, train_images_file_name = tisu.random_image_set_up() 
    train_images = tisu.image_flatten(train_images_file_name, resize)
    #  重みとバイアスの準備  
    wb = weight_bias_set_up(weight_composition)
    if os.path.exists(file_name):
        print('weight load')
        w, b = wb.weight_load(file_name)
    else:
        print('weight create')
        w, b = wb.weight_create(weight_composition) 
    #　学習の準備
    def train_start(true_labels, train_images, w, b, hyper_parm_composition, file_name):
        isinstance_propagation = propagation(net_composition)
        true_count = 0
        for i in range(0, len(train_images)):
            #順伝播
            forward_x_cache, forward_activation_cache, forward_dropout_cache, transmission = isinstance_propagation.forward_start(
                train_images[i],
                w, 
                b, 
                hyper_parm_composition["dropout"]
            )  
            #評価と損失の出力
            true_count += isinstance_propagation.aggregate(
                transmission, 
                true_labels[i]
            )
            loss = isinstance_propagation.evaluation_and_cost(
                transmission,
                true_labels[i]
            )
            #逆伝播
            update_delta = isinstance_propagation.backward_start(
                loss,
                forward_activation_cache,
                forward_dropout_cache, 
                w
            )
            #重みとバイアスの更新
            isinstance_propagation.weight_and_bias_update(
                w,
                b, 
                hyper_parm_composition["learning_rate"],
                forward_x_cache,
                update_delta)
            isinstance_propagation.weight_and_bias_save(
                file_name,
                w,
                b,
                weight_composition["layer-length"]
            )
        print(f'successfull training: {true_count/train_set_size}')
    def test_model(true_labels, train_images, net_composition, w, b):
        true_count = 0
        for j in range(0, len(train_images)):
            transmission = train_images[j].reshape((1,len(train_images[j])))
            for i in range(0, len(net_composition)):
                transmission = np.dot(transmission, w[f"w{i+1}"]) + b[f"b{i+1}"]
                transmission = activation_functions.activation(net_composition[i]["activation"],transmission)
            solve_index = np.argmax(transmission)
            true_index = np.argmax(true_labels[j])
            if solve_index == true_index:
                true_count += 1       
            # else:
            #     print("___A___")
            #     img = Image.open(train_images_file_name[j])
            #     img.show()
            print(transmission,true_labels[j])
        print(f'successfull testing: {true_count/train_set_size}')    
    if train:
        train_start(true_labels, train_images, w, b, hyper_parm_composition, file_name)
    if test:
        test_model(true_labels, train_images, net_composition, w, b)
nn_execute(file_name, hyper_parm_composition, net_composition, train_set_size, resize)




