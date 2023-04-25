from scipy.signal import resample_poly
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import correlate, correlation_lags
from os.path import join as pjoin
import os


def generate(u=1, seq_length=813, q=0, dtype=np.complex64):
    """
    Generate a Zadoff-Chu (ZC) sequence.
    Parameters
    ----------
    u : int
        Root index of the the ZC sequence: u>0.
    seq_length : int
        Length of the sequence to be generated. Usually a prime number:
        u<seq_length, greatest-common-denominator(u,seq_length)=1.
    q : int
        Cyclic shift of the sequence (default 0).
    Returns
    -------
    zcseq : 1D ndarray of complex floats
        ZC sequence generated.
        
   To still put DC to 0, 
   we interleave the ZC sequence with zeros.
 
    """

    for el in [u, seq_length, q]:
        if not float(el).is_integer():
            raise ValueError('{} is not an integer'.format(el))
    if u <= 0:
        raise ValueError('u is not stricly positive')
    if u >= seq_length:
        raise ValueError('u is not stricly smaller than seq_length')

    if np.gcd(u, seq_length) != 1:
        raise ValueError('the greatest common denominator of u and seq_length is not 1')

    cf = seq_length % 2
    n = np.arange(seq_length)
    zcseq = np.exp(-1j * np.pi * u * n * (n+cf+2.0*q) / seq_length, dtype=dtype)

    return zcseq


dt = np.dtype([('re', np.int16), ('im', np.int16)])


NZC = 813  
num_samples = 1024

zc_fft = generate(u=1, seq_length=NZC)
zc_time = None


files = ["usrp_samples_31DBE03_0.dat","usrp_samples_31DBE03_1.dat","usrp_samples_31DEA71_0.dat","usrp_samples_31DEA71_1.dat"]
dirname = os.path.dirname(__file__)


# y = IQ time
# y_peaks = normalize(conv(y, ZC_time))
# y_zc = extract NZC at each y_peaks > 0.9
# for each y_zc
# dpd = ifft(fft(y_zc) / ZC_Freq)

num_channels = 4
channels = range(num_channels)


# Load the IQ samples from the stored files for each channel
IQ_matrix = []
for i in range(num_channels):
    #x = np.fromfile(pjoin(dirname, files[i]), dtype=dt)
    x = np.fromfile(pjoin(dirname, "results", "1e6Sps", files[i]), dtype=dt)
    samples = np.zeros(len(x),dtype=np.complex64)

    samples.real = x['re']/(2**15)
    samples.imag= x['im']/(2**15)
    print(len(samples))

    
    IQ_matrix.append(samples)

IQ_matrix = np.atleast_2d(IQ_matrix) # for instance if there is only one channel
_, total_samples = IQ_matrix.shape

print(IQ_matrix.shape)

start_idx = int(1.6*1e6) #start after 1.5 seconds
num_sequences = (total_samples - start_idx)//num_samples
num_sequences = min(num_sequences, 10)
IQ_matrix = IQ_matrix[:, start_idx:start_idx+num_samples*num_sequences]

a = np.zeros_like(samples, shape=(num_channels, num_sequences, num_samples))

for ch in channels:
    splitted = np.asarray(np.split(IQ_matrix[ch,:], num_sequences))
    a[ch,:,:] = splitted

yf = np.fft.fft(a,axis=-1)
yf = np.roll(yf, NZC//2, axis=-1)[:, :, :NZC]


for i in range(num_sequences):
    plt.plot(20*np.log10(np.abs(np.fft.ifft(yf[0,i,:]/zc_fft))))
plt.show()


for ch in channels:
    plt.plot(20*np.log10(np.abs(yf[ch,0,:]/zc_fft)))
plt.show()


# CFO 

# norm = np.linalg.norm(yf,axis=-1) * np.linalg.norm(yf[:, 0], axis=-1)

# alpha = np.conjugate(yf) * yf[:,0:1]
# alpha /= np.asarray([np.repeat(norm[i], NZC) for i in range(num_channels)])

# delta_f = 1e6/num_samples
# print(delta_f)
# for i in range(num_channels):
#     plt.plot(np.angle(alpha[i,:]))
#     expected_phase = np.exp([2.0*np.pi*1j* delta_f * t for t in range(alpha.shape[-1])])
#     plt.plot(np.angle(expected_phase))
#     plt.show()


h_abs = 20*np.log10(np.abs(np.fft.ifft(yf/zc_fft, axis=1)))

# IQ_matrix = IQ_matrix[:,total_samples//2:(total_samples//2)+NZC]
# y_fft = np.fft.fft(IQ_matrix, axis=-1)

# h_fft = y_fft/np.fft.fftshift(zc_fft)

# h_abs = 20*np.log10(np.abs(np.fft.ifft(h_fft)))

#axes = axes if isinstance(axes, list) else [axes]

for i in range(num_channels):
    plt.plot(h_abs[i, :], label=f"Channel {i}")
plt.legend()
plt.show()


h_abs = 20*np.log10(np.abs(np.fft.fftshift(y)))

fig, axes = plt.subplots(num_channels)
axes = axes if isinstance(axes, list) else [axes]
for i in range(num_channels):
    axes[i].plot(h_abs[i, :])
    axes[i].plot((np.abs(np.fft.fftshift(zc_fft))))
plt.show()


fig, axes = plt.subplots(num_channels)
axes = axes if isinstance(axes, list) else [axes]
y_fft = np.roll(y_fft, NZC//2)
y = 20*np.log10(abs(np.fft.ifft(y_fft/zc_fft)))
yidx = np.argmax(y)
y = y * np.exp(1j*2*np.pi*yidx/NZC*np.arange(NZC))
y = np.angle(y/zc_fft)

for i in range(num_channels):
    axes[i].plot(y[i, :])
plt.show()
