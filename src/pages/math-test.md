---
title: "Math Test Page"
description: "Testing mathematical formula rendering"
---

# 数学公式测试页面

## 行内公式测试

这是一个行内公式：$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$，它应该能正确渲染。

## 块级公式测试

这是一个块级公式：

$$
\begin{align}
E &= mc^2 \\
F &= ma \\
\nabla \cdot \mathbf{E} &= \frac{\rho}{\epsilon_0}
\end{align}
$$

## Diffusion Model 公式测试

扩散模型中的重要公式：

$$q(x_t|x_0) = N(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)I)$$

其中 $\bar{\alpha}_t$ 是由每一步的噪声水平 $\beta_s$ (通过 $\alpha_s = 1-\beta_s$) 累乘得到的。

## 复杂公式测试

一个更复杂的公式：

$$
\mathcal{L} = \mathbb{E}_{t,x_0,\epsilon}\left[\left\|\epsilon - \epsilon_\theta\left(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t\right)\right\|^2\right]
$$

这些公式应该都能正确渲染。
