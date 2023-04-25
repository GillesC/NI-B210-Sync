#!/usr/bin/python3
import sys
import numpy as np

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
from scipy.signal import resample_poly

from os import listdir
from os.path import isfile, join, isdir

from os.path import join as pjoin


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


files = ["usrp_samples_31DBE03_0.dat", "usrp_samples_31DBE03_1.dat", "usrp_samples_31DEA71_0.dat", "usrp_samples_31DEA71_1.dat"]
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
    samples = np.zeros(len(x), dtype=np.complex64)

    samples.real = x['re']/(2**15)
    samples.imag = x['im']/(2**15)
    print(len(samples))

    IQ_matrix.append(samples)

IQ_matrix = np.atleast_2d(IQ_matrix)  # for instance if there is only one channel
_, total_samples = IQ_matrix.shape

print(IQ_matrix.shape)

start_idx = int(1.6*1e6)  # start after 1.5 seconds
num_sequences = (total_samples - start_idx)//num_samples
num_sequences = min(num_sequences, 10)
IQ_matrix = IQ_matrix[:, start_idx:start_idx+num_samples*num_sequences]

iq_samples = np.zeros_like(samples, shape=(num_channels, num_sequences, num_samples))

for ch in channels:
    splitted = np.asarray(np.split(IQ_matrix[ch, :], num_sequences))
    iq_samples[ch, :, :] = splitted

yf = np.fft.fft(iq_samples, axis=-1)
yf = np.roll(yf, NZC//2, axis=-1)[:, :, :NZC]

app = Dash(__name__)

app.layout = html.Div(children=[
    dcc.Dropdown(['live','not live'], 'live', id='dropdown2'),
    dcc.Dropdown(['pdp','freq','phase'], 'pdp', id='dropdown'),
    dcc.Graph(id='graph'),
    dcc.Input(
        id='input-text',
        type='text',
        value='.'
    ),
    dcc.Input(
        id='input-text2',
        type='text',
        value=''
    ),
    dcc.Input(
        id='input-number',
        type='number',
        value=0
    ),
    dcc.Interval(
        id='interval-component',
        interval=1000,
        n_intervals=0
    )
])


@app.callback([
              Output('graph', 'figure'),
              Output('input-number','value'),
              Output('input-text','value'),
              Output('input-text2','value')
              ],
              [
               Input('input-number', 'value'),
               Input('input-text','value'),
               Input('dropdown','value'),
               Input('interval-component','n_intervals'),
               Input('dropdown2','value'),
               Input('input-text2','value')
               ])
def update_graph(*vals):
    print(vals)
    choice = vals[2]
    upsamp = 1 # only for visibility

    fig = go.Figure()


    for ch in range(num_channels):
        # get the file
        if choice == 'pdp':
            
            y = 20*np.log10(abs(np.fft.ifft(yf/zc_fft)))
            y = resample_poly(y, upsamp, 1)


            # make graph
            fig.add_trace(go.Scatter(x=np.arange(NZC),
                    y=y, mode='lines',name=f'ch{ch}'))

    if choice == 'pdp':
        fig.update_layout(xaxis_title='Delay', yaxis_title='Channel gain [dB]', height=550)
    elif choice == 'phase':
        fig.update_layout(yaxis_range=[-3.14,3.14],xaxis_title='Frequency', yaxis_title='Phase [rad]', height=550)
    elif choice == 'freq':
        fig.update_layout(yaxis_range=[-3.14,3.14],xaxis_title='Frequency', yaxis_title='Channel gain [dB]', height=550)

    if vals[4] == 'live':
        fig['layout']['uirevision'] = 'interval-component'
    else:
        fig['layout']['uirevision'] = 'interval-number'

    return fig


if __name__ == '__main__':
    app.run_server(host='0.0.0.0',debug=True, port=8050)
