import numpy as np

class MulLayerGeneral:
    def __init__(self):
        # 入力値を保存しておくためのリスト
        self.inputs = None

    def forward(self, *inputs):
        """
        順伝播: 引数をいくつでも受け取り、すべての積を返す
        例: forward(a, b, c) -> out = a * b * c
        """
        self.inputs = inputs # 逆伝播のためにタプルとして保存
        
        out = 1.0
        for x in inputs:
            out = out * x
            
        return out

    def backward(self, dout):
        """
        逆伝播: 各入力に対して「自分以外の全要素の積」に dout を掛けたものを返す
        """
        dx_list = []
        
        # 入力の数だけループ（x, y, z... それぞれの勾配を計算）
        for i in range(len(self.inputs)):
            
            # --- ここが「自分以外を掛ける」ロジック ---
            product_of_others = 1.0
            for j, x in enumerate(self.inputs):
                if i != j:  # インデックスが自分(i)じゃなければ掛ける
                    product_of_others = product_of_others * x
            # ---------------------------------------

            # 勾配 = dout * (自分以外の積)
            dx_list.append(dout * product_of_others)
            
        return tuple(dx_list)

# --- 動作確認 ---

# 1. インスタンス生成
layer = MulLayerGeneral()

# 2. 順伝播 (x=2, y=3, z=4) -> 2*3*4 = 24
x, y, z = 2.0, 3.0, 4.0
out = layer.forward(x, y, z)
print(f"Forward Result: {out}") # -> 24.0

# 3. 逆伝播 (dout=1 とする)
# 期待値:
# dx = 1 * (3 * 4) = 12
# dy = 1 * (2 * 4) = 8
# dz = 1 * (2 * 3) = 6
dx, dy, dz = layer.backward(dout=1.0)

print(f"Backward Result: dx={dx}, dy={dy}, dz={dz}")
