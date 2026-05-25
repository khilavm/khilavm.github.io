---
layout: post
title: The Danielson-Lanczos FFT method and bit-reversal
date: 2025-12-01 00:00:00-0600
description: A walkthrough of the Danielson-Lanczos method for computing the Fast Fourier Transform, including the bit-reversal permutation.
tags: math fft
categories: notes
related_posts: false
author: Khilav Majmudar
---

This note summarizes my understanding of the Danielson-Lanczos method of calculating the Fast Fourier Transform of an array of numbers. I read it in Numerical Recipes[^1]. The method explained there becomes clear, as in most such cases, by expanding out the expressions for small values. The Danielson-Lanczos method is the prototype example of the general class of FFT algorithms.

We have a discrete Fourier transform of length $$N$$.

\begin{equation}\label{main}
    F_k = \sum_{j=0}^{N-1} e^{2 \pi i j k/N} f_{j}
\end{equation}

For now, we will only consider cases in which $$N$$ is a power of 2 as this is where the approach is most clearly seen. Let us split this according to its even and odd components.

$$
\begin{align}
    F_k &= \sum_{j=0}^{N-1} e^{2 \pi i j k/N} f_{j} \nonumber \\
    &= \sum_{j=0}^{N/2 - 1} e^{2 \pi i (2j) k/N} f_{2j}+\sum_{j=0}^{N/2 - 1} e^{2 \pi i (2j+1) k/N} f_{2j+1} \nonumber \\
    &= \sum_{j=0}^{N/2 - 1} e^{2 \pi ij k/(N/2)} f_{2j}+ W^{k} \sum_{j=0}^{N/2 - 1} e^{2 \pi ij k/(N/2)} f_{2j+1} \nonumber \\
    &= F_{k}^{e} + W^{k}F_{k}^{o} \label{eq1}
\end{align}
$$

Here, $$F_k$$ is the $$k$$-th component of the Fourier-transformed array. The sum that leads to it has been split into even and odd transforms of length $$N/2$$, given by $$F_{k}^{e}$$ and $$F_{k}^{o}$$ respectively. $$W$$ is a constant factor and is given by $$e^{2\pi i/N}$$. Now suppose that $$N=2$$. Then the sums in the second-last line collapse to one element each and are given by

\begin{equation}\label{eq2}
    F_k = f_0 + W^{k}f_1
\end{equation}

This is our DFT. For completeness' sake, the elements of $$F$$ are

$$
\begin{align*}
    & F_0=f_0+W^0f_1=f_0+f_1 \\
    & F_1=f_0+W^1f_1=f_0+e^{i\pi} f_1=f_0-f_1
\end{align*}
$$

Let us take $$N=4$$. Then we can go further and create four Fourier transforms of length $$N/4$$. We start from the second-last line of equation \eqref{eq1} and split each term into even and odd parts.

$$
\begin{align}
    F_k &= \sum_{j=0}^{N/2 - 1} e^{2 \pi ij k/(N/2)} f_{2j}+ W^{k} \sum_{j=0}^{N/2 - 1} e^{2 \pi ij k/(N/2)} f_{2j+1} \nonumber \\
    &= \sum_{j=0}^{N/4 - 1} e^{2 \pi i (2j) k/(N/2)} f_{2(2j)} + \sum_{j=0}^{N/4 - 1} e^{2 \pi i (2j+1) k/(N/2)} f_{2(2j+1)} \nonumber \\
    &\quad + W^{k} \sum_{j=0}^{N/4 - 1} e^{2 \pi i(2j)k/(N/2)} f_{2(2j)+1} \nonumber \\
    &\quad + W^{k} \sum_{j=0}^{N/4 - 1} e^{2 \pi i(2j+1)k/(N/2)} f_{2(2j+1)+1} \nonumber \\
    &= \sum_{j=0}^{N/4 - 1} e^{2 \pi ij k/(N/4)} f_{4j} + W^{2k} \sum_{j=0}^{N/4 - 1} e^{2 \pi ij k/(N/4)} f_{4j+2} \nonumber \\
    &\quad + W^{k} \sum_{j=0}^{N/4 - 1} e^{2 \pi ij k/(N/4)} f_{4j+1} + W^{k} W^{2k} \sum_{j=0}^{N/4 - 1} e^{2 \pi ij k/(N/4)} f_{4j+3} \nonumber \\
    &= F_{k}^{ee}+W^{2k}F_{k}^{eo}+W^{k}F_{k}^{oe}+W^{3k}F_{k}^{oo} \nonumber \\
    &= f_{0} +W^{2k}f_{2} +W^{k}f_{1} +W^{3k}f_{3} \label{eq3}
\end{align}
$$

Notice how the power of $$W$$ is the same as the index of the element of the original array $$f$$ multiplying it. Also, $$W^{N}=W^{0}=1$$. For $$N=4$$, $$W=e^{i\pi/2}$$. The complete $$F$$ is as follows:

$$
\begin{align*}
    & F_0=f_0+f_2+f_1+f_3 \\
    & F_1=f_0-f_2+if_1-if_3 \\
    & F_2=f_0+f_2-f_1-f_3 \\
    & F_3=f_0-f_2-if_1+if_3
\end{align*}
$$

Let us do this one more time for $$N=8$$.

$$
\begin{align}
    F_k &= \sum_{j=0}^{N/8 - 1} e^{2 \pi ij k/(N/8)} f_{8j} + W^{4k} \sum_{j=0}^{N/8 - 1} e^{2 \pi ij k/(N/8)} f_{8j+4} \nonumber \\
    &\quad + W^{2k} \sum_{j=0}^{N/8 - 1} e^{2 \pi ij k/(N/8)} f_{8j+2} + W^{2k} W^{4k} \sum_{j=0}^{N/8 - 1} e^{2 \pi ij k/(N/8)} f_{8j+6} \nonumber \\
    &\quad + W^{k} \sum_{j=0}^{N/8 - 1} e^{2 \pi ij k/(N/8)} f_{8j+1} + W^{k} W^{4k} \sum_{j=0}^{N/8 - 1} e^{2 \pi ij k/(N/8)} f_{8j+5} \nonumber \\
    &\quad + W^{k}W^{2k} \sum_{j=0}^{N/8 - 1} e^{2 \pi ij k/(N/8)} f_{8j+3} + W^{k}W^{2k}W^{4k} \sum_{j=0}^{N/8 - 1} e^{2 \pi ij k/(N/8)} f_{8j+7} \nonumber \\
    &= F_{k}^{eee}+W^{4k}F_{k}^{eeo}+W^{2k}F_{k}^{eoe}+W^{6k}F_{k}^{eoo} \nonumber \\
    &\quad + W^{k}F_{k}^{oee}+W^{5k}F_{k}^{oeo}+W^{3k}F_{k}^{ooe}+W^{7k}F_{k}^{ooo} \nonumber \\
    &= f_{0} +W^{4k}f_{4} +W^{2k}f_{2} +W^{6k}f_{6} + W^{k}f_{1} +W^{5k}f_{5} +W^{3k}f_{3} +W^{7k}f_{7} \label{eq4}
\end{align}
$$

Through three decompositions in halves, we have reduced our DFT calculation to 8 one-point transforms. The bookkeeping of which element in $$f$$ corresponds to which odd-even subdivision is possible for small $$N$$'s, but it quickly becomes cumbersome. In equation \eqref{eq4} above, it is easy to go from the first line to the third, but to get to line 3 from line 2 is the point of *bit-reversal*.

Take the $$eeoeoeo\ldots oo$$ pattern, reverse it and assign 1 to $$o$$ and 0 to $$e$$. This is the value in binary of the element in $$f$$ whose one-point Fourier transform is represented as $$F_{k}^{eeoeoeo\ldots oo}$$. For example, $$eoo \mapsto 110 = 6$$. Thus, the bit-reversed data array $$f$$, in index terms, for $$N=8$$, would be $$\{0,4,2,6,1,5,3,7\}$$. These bit-reversed elements are then combined in pairs (going backward in our equations) until finally the combined sum of halves results in the final FFT. So, the basic idea of an FFT algorithm is to first bit-reverse the original data by swapping elements and then to compute the subtransforms of length $$2,4,8,\ldots,N$$.

To understand the above, take the original indices of $$f$$ for $$N=4$$ --- $$\{0,1,2,3\}$$ --- in binary, i.e., $$\{00,01,10,11\}$$. We can see that $$f_0$$ and $$f_3$$ bit-reverse to themselves, and $$f_1$$ and $$f_2$$ map to each other. Thus the bit-reversed array is $$\{0,2,1,3\}$$. For $$N=8$$, we have the indices of $$f$$ in binary as $$\{000,001,010,011,100,101,110,111\}$$. Here, $$f_0$$, $$f_2$$, $$f_5$$ and $$f_7$$ map to themselves, and $$f_1 \leftrightarrow f_4$$, $$f_3 \leftrightarrow f_6$$. This leads to the bit-reversed array given in the previous paragraph.

Suppose we wanted to construct the bit-reversed index array for $$N=16$$. We would have to write the binary representation of every index, reverse it and then get the resulting array. This process quickly becomes cumbersome. But there is an easier way, inspired by the pseudocode presented in Johnson & Frigo[^2]. Consider the bit-reversed arrays for $$N \in \{2,4,8\}$$.

$$
\begin{align*}
    & N=2 \implies \{0,1\} \\
    & N=4 \implies \{0,2,1,3\} \\
    & N=8 \implies \{0,4,2,6,1,5,3,7\}
\end{align*}
$$

Do we see a pattern here? To construct the bit-reversed array for $$N=p$$, take the one for $$N=p/2$$ and after each element $$i$$, insert an index $$i+p/2$$. We will rewrite the arrays given above but with the new elements in red. The elements in black are carried over from the previous $$N$$. The $$N=16$$ case is also shown.

$$
\begin{align*}
    & N=2 \implies \{0,1\} \\
    & N=4 \implies \{0,{\color{red}{2}},1,{\color{red}{3}}\} \\
    & N=8 \implies \{0,{\color{red}{4}},2,{\color{red}{6}},1,{\color{red}{5}},3,{\color{red}{7}}\} \\
    & N=16 \implies \{0,{\color{red}{8}},4,{\color{red}{12}},2,{\color{red}{10}},6,{\color{red}{14}},1,{\color{red}{9}},5,{\color{red}{13}},3,{\color{red}{11}},7,{\color{red}{15}}\}
\end{align*}
$$

This halving algorithm achieves a significant reduction in computational time complexity. Evaluating the DFT directly using equation \eqref{main} requires $$N^2$$ complex multiplications, which is represented by a complexity of $$\mathcal{O}(N^2)$$. As $$N$$ becomes large — which is typical in audio processing, image analysis, and scientific computing — this quadratic growth becomes computationally prohibitive. In the Danielson-Lanczos method, we have $$\log_2(N)$$ halvings of the data array and $$N$$ operations to combine the subtransforms at each stage, giving us an $$\mathcal{O}(N \log_2 N)$$ process. To illustrate the impact of this difference, for a dataset of one million points ($$N=10^6$$), the direct $$\mathcal{O}(N^2)$$ approach requires a trillion operations, whereas the FFT requires only about 20 million --- a speedup by a factor of fifty thousand.

[^1]: W. H. Press, *Numerical Recipes in Pascal: The Art of Scientific Computing*, Cambridge University Press, 1989.
[^2]: S. G. Johnson and M. Frigo, "Implementing FFTs in practice," arXiv:2602.23525, 2026.
