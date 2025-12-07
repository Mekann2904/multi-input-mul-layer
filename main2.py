class MulLayerOptimized:
    def __init__(self):
        self.inputs = None

    def forward(self, *inputs):
        self.inputs = inputs
        out = 1.0
        for x in inputs:
            out *= x
        return out

    def backward(self, dout):
        inputs = self.inputs
        n = len(inputs)
        
        # 1. ゼロがどこにあるか探す（これだけで後の計算方針が決まる）
        zero_indices = [i for i, x in enumerate(inputs) if x == 0]
        zero_count = len(zero_indices)
        
        # 結果を格納するリスト（初期値0で用意）
        dx = [0.0] * n

        # --- パターンA: ゼロがない場合 (通常ルート) ---
        if zero_count == 0:
            # 全体の積を再計算
            total_product = 1.0
            for x in inputs:
                total_product *= x
            
            # 割り算テクニックで高速計算
            for i in range(n):
                # 自分以外の積 = 全体の積 / 自分
                dx[i] = dout * (total_product / inputs[i])

        # --- パターンB: ゼロが1個だけある場合 ---
        elif zero_count == 1:
            # ゼロの要素以外の勾配はすべて0になる（なぜなら積に0が含まれるから）
            # ゼロの要素自身の勾配だけ計算すればよい
            
            zero_idx = zero_indices[0] # ゼロの場所
            
            # ゼロ以外の要素をすべて掛け合わせる
            product_without_zero = 1.0
            for i, x in enumerate(inputs):
                if i != zero_idx:
                    product_without_zero *= x
            
            dx[zero_idx] = dout * product_without_zero

        # --- パターンC: ゼロが2個以上ある場合 ---
        else:
            # どう組み合わせても「自分以外」の中に必ずゼロが1個以上残るため、
            # 全員の勾配が 0 になる。
            # (dxは初期化で既に [0.0, 0.0...] なので何もしなくてOK)
            pass

        return tuple(dx)
