def main():
    import argparse

    import numpy
    try:
        import cupy
        available = True
    except ImportError:
        available = False
    import librosa

    from PCA import RPCA

    if available and args.gpu >= 0:
        cupy.cuda.Device(args.gpu).use()
        xp = cupy
        available = True
    else:
        xp = numpy
        available = False

    accmp,sr = librosa.load("music.wav", 16000, mono=False)
    vocal,sr2 = librosa.load("vocal.wav", 16000, mono=False)
    print(accmp)
    print(sr)

    mixed = 0.5 * (vocal + accmp)
    M = librosa.stft(mixed, n_fft=1024, hop_length=512)
    phase = numpy.angle(M)
    M = xp.abs(xp.expand_dims(xp.asarray(M), 2))
    print('shape: {}, min: {}, max: {}'.format(M.shape, M.min(), M.max()))

    lmd = 0.03
    rho = 0.8
    max_iter = 1000
    stopcri = 0.05

    rpca = RPCA(xp, lmd, rho, max_iter, stopcri)
    L, S = rpca(M, echo_iter=5)

    M = xp.maximum(M, 0)
    L = xp.maximum(L, 0)
    S = xp.maximum(S, 0)

    if available:
        M = xp.asnumpy(M)
        L = xp.asnumpy(L)
        S = xp.asnumpy(S)

    L = numpy.squeeze(L) * numpy.exp(1j*phase)
    S = numpy.squeeze(S) * numpy.exp(1j*phase)
    M_recon = numpy.squeeze(L + S) * numpy.exp(1j*phase)
    M = numpy.squeeze(M) * numpy.exp(1j*phase)

    librosa.output.write_wav(
        #'L4.wav', librosa.istft(L, hop_length=512), 16000)
        'Low_rank.wav', librosa.istft(L, hop_length=512), 16000)
    librosa.output.write_wav(
        #'S4.wav', librosa.istft(S, hop_length=512), 16000)
        'Sparse_Vocal.wav', librosa.istft(S, hop_length=512), 16000)
    librosa.output.write_wav(
      #  'M_recon4.wav', librosa.istft(M_recon, hop_length=512),
       # 16000)
    'M_reconstructed.wav', librosa.istft(M_recon, hop_length=512),
        16000)
    librosa.output.write_wav(
       # 'M4.wav', librosa.istft(M, hop_length=512), 16000)
        'M_original.wav', librosa.istft(M, hop_length=512), 16000)


if __name__ == '__main__':
    main()
