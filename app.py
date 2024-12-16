import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
import numpy as np
import tensorflow as tf
from datetime import datetime
import os
import threading
from tensorflow.keras.callbacks import Callback

class TrainingCallback(Callback):
    def __init__(self, progress_var, status_text):
        super().__init__()
        self.progress_var = progress_var
        self.status_text = status_text

    def on_epoch_end(self, epoch, logs=None):
        # 更新进度条和状态文本
        progress = int(((epoch + 1) / self.params['epochs']) * 100)
        self.progress_var.set(progress)
        self.status_text.set(f"训练中... 第 {epoch + 1}/{self.params['epochs']} 轮 "
                           f"损失: {logs['loss']:.4f}")
        
class LotteryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("双色球预测系统 v2.0")
        self.root.geometry("1000x800")
        
        # 设置样式
        style = ttk.Style()
        style.configure('Custom.TButton', padding=5)
        
        # 初始化变量
        self.model = None
        self.sequence_length = 20
        self.status_text = tk.StringVar()
        self.progress_var = tk.DoubleVar()
        self.file_path = tk.StringVar()
        self.epochs_var = tk.StringVar(value="50")
        self.batch_size_var = tk.StringVar(value="32")
        
        self.create_gui()
        self.create_model()

    def create_model(self):
        try:
            input_layer = tf.keras.Input(shape=(self.sequence_length, 6))
            x = tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu')(input_layer)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu')(x)
            x = tf.keras.layers.GlobalMaxPooling1D()(x)
            x = tf.keras.layers.Dense(128, activation='relu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            x = tf.keras.layers.Dense(64, activation='relu')(x)
            
            red_output = tf.keras.layers.Dense(33, activation='sigmoid', name='red_output')(x)
            blue_output = tf.keras.layers.Dense(16, activation='sigmoid', name='blue_output')(x)
            
            self.model = tf.keras.Model(inputs=input_layer, outputs=[red_output, blue_output])
            self.model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss={'red_output': 'binary_crossentropy', 
                      'blue_output': 'binary_crossentropy'},
                metrics=['accuracy']
            )
        except Exception as e:
            messagebox.showerror("错误", f"模型创建失败: {str(e)}")

    def create_gui(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 文件选择区域
        file_frame = ttk.LabelFrame(main_frame, text="数据选择", padding="5")
        file_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(file_frame, text="历史数据文件:").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(file_frame, textvariable=self.file_path, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="浏览", command=self.browse_file).grid(row=0, column=2)
        
        # 参数设置区域
        param_frame = ttk.LabelFrame(main_frame, text="训练参数", padding="5")
        param_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(param_frame, text="训练轮数:").grid(row=0, column=0, padx=5)
        ttk.Entry(param_frame, textvariable=self.epochs_var, width=10).grid(row=0, column=1)
        
        ttk.Label(param_frame, text="批次大小:").grid(row=0, column=2, padx=5)
        ttk.Entry(param_frame, textvariable=self.batch_size_var, width=10).grid(row=0, column=3)
        
        # 控制按钮
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        ttk.Button(button_frame, text="开始训练", 
                  command=lambda: threading.Thread(target=self.train).start(),
                  style='Custom.TButton').grid(row=0, column=0, padx=5)
        ttk.Button(button_frame, text="开始预测", 
                  command=self.predict,
                  style='Custom.TButton').grid(row=0, column=1, padx=5)
        
        # 进度显示
        progress_frame = ttk.Frame(main_frame)
        progress_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(progress_frame, textvariable=self.status_text).grid(row=0, column=0, sticky=tk.W)
        self.progress = ttk.Progressbar(progress_frame, length=300, mode='determinate', 
                                      variable=self.progress_var)
        self.progress.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # 结果显示区域
        result_frame = ttk.LabelFrame(main_frame, text="预测结果", padding="5")
        result_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        
        self.result_text = tk.Text(result_frame, height=20, width=80)
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.result_text['yscrollcommand'] = scrollbar.set

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="选择Excel文件",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        if filename:
            self.file_path.set(filename)
            self.status_text.set(f"已选择文件: {os.path.basename(filename)}")

    def prepare_data(self, data):
        sequences = []
        red_labels = []
        blue_labels = []
        
        for i in range(len(data) - self.sequence_length):
            seq = data.iloc[i:i + self.sequence_length, 1:7].values.astype(np.float32)
            sequences.append(seq)
            
            red_label = np.zeros(33, dtype=np.float32)
            for num in data.iloc[i + self.sequence_length, 1:7]:
                red_label[int(num) - 1] = 1
            red_labels.append(red_label)
            
            blue_label = np.zeros(16, dtype=np.float32)
            blue_label[int(data.iloc[i + self.sequence_length, 7]) - 1] = 1
            blue_labels.append(blue_label)
            
        return np.array(sequences), np.array(red_labels), np.array(blue_labels)

    def train(self):
        try:
            if not self.file_path.get():
                messagebox.showerror("错误", "请先选择数据文件！")
                return
                
            self.status_text.set("正在加载数据...")
            data = pd.read_excel(self.file_path.get())
            
            self.status_text.set("正在准备训练数据...")
            X, y_red, y_blue = self.prepare_data(data)
            
            epochs = int(self.epochs_var.get())
            batch_size = int(self.batch_size_var.get())
            
            self.status_text.set("开始训练...")
            self.progress_var.set(0)
            
            callback = TrainingCallback(self.progress_var, self.status_text)
            
            history = self.model.fit(
                X, 
                {'red_output': y_red, 'blue_output': y_blue},
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[
                    callback,
                    tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                    tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
                ]
            )
            
            self.status_text.set("训练完成！")
            self.progress_var.set(100)
            
            # 显示训练结果
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "训练完成！\n")
            self.result_text.insert(tk.END, f"最终损失: {history.history['loss'][-1]:.4f}\n")
            self.result_text.insert(tk.END, f"验证损失: {history.history['val_loss'][-1]:.4f}\n")
            
        except Exception as e:
            self.status_text.set("训练失败！")
            messagebox.showerror("错误", f"训练过程中出错: {str(e)}")

    def predict(self):
        try:
            if not self.file_path.get():
                messagebox.showerror("错误", "请先选择数据文件！")
                return
                
            if self.model is None:
                messagebox.showerror("错误", "请先训练模型！")
                return
            
            data = pd.read_excel(self.file_path.get())
            
            # 准备预测数据
            last_sequence = data.iloc[-self.sequence_length:, 1:7].values.astype(np.float32)
            input_data = np.expand_dims(last_sequence, axis=0)
            
            # 预测
            predictions = self.model.predict(input_data)
            
            # 获取前10个红球和4个蓝球的概率最高的号码
            red_predictions = np.argsort(predictions[0][0])[-10:] + 1
            blue_predictions = np.argsort(predictions[1][0])[-4:] + 1
            
            # 计算概率
            red_probs = predictions[0][0][red_predictions - 1]
            blue_probs = predictions[1][0][blue_predictions - 1]
            
            # 显示结果
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, f"预测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            self.result_text.insert(tk.END, "推荐红球号码及概率:\n")
            for num, prob in zip(sorted(red_predictions), red_probs[np.argsort(red_predictions)]):
                self.result_text.insert(tk.END, f"号码 {num:02d}: {prob*100:.2f}%\n")
            
            self.result_text.insert(tk.END, "\n推荐蓝球号码及概率:\n")
            for num, prob in zip(sorted(blue_predictions), blue_probs[np.argsort(blue_predictions)]):
                self.result_text.insert(tk.END, f"号码 {num:02d}: {prob*100:.2f}%\n")
            
        except Exception as e:
            messagebox.showerror("错误", f"预测过程中出错: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = LotteryApp(root)
    root.mainloop()
