from .C2_Complex import generate_figure, \
    plot_vector

from .C2_Fourier import generate_matrix_dft, \
    generate_matrix_dft_inv, \
    dft, \
    idft, \
    fft, \
    ifft_noscale, \
    ifft, \
    stft_basic, \
    istft_basic, \
    stft, istft, \
    stft_conventionFMP

from .C2_Interpolation import compute_F_coef_linear, \
    compute_F_coef_log, \
    interpolate_freq_stft

from .C2_Interference import plot_interference, \
    generate_chirp_linear

from .C2_Digitization import generate_function, \
    sampling_equidistant, \
    reconstruction_sinc, \
    plot_graph_quant_function, \
    quantize_uniform, \
    plot_signal_quant, \
    encoding_mu_law, \
    decoding_mu_law, \
    plot_mu_law, \
    quantize_nonuniform_mu
