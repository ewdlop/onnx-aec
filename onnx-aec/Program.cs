using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NAudio.Wave;
using System.Collections.Concurrent;
using System.Numerics;
using MathNet.Numerics;
using MathNet.Numerics.IntegralTransforms;

namespace EchoCancellation
{
    public class DECModelRealTime : IDisposable
    {
        private readonly InferenceSession _model;
        private readonly float _hopFraction;
        private readonly int _dftSize;
        private readonly int _hiddenSize;
        private readonly int _samplingRate;
        private readonly int _frameSize;
        private readonly float[] _window;
        private float[,,] _h01;
        private float[,,] _h02;

        public DECModelRealTime(string modelPath, double windowLength = 0.02, float hopFraction = 0.5f,
            int dftSize = 320, int hiddenSize = 322, int samplingRate = 16000)
        {
            _hopFraction = hopFraction;
            _dftSize = dftSize;
            _hiddenSize = hiddenSize;
            _samplingRate = samplingRate;
            _frameSize = (int)(windowLength * samplingRate);

            // Initialize Hamming window (equivalent to np.sqrt(np.hamming(frame_size)))
            _window = CreateHammingWindow(_frameSize);

            Console.WriteLine($"Model initialized with frame_size: {_frameSize}, window length: {_window.Length}");

            try
            {
                if (!File.Exists(modelPath))
                {
                    throw new FileNotFoundException($"Model file not found: {modelPath}");
                }

                var sessionOptions = new SessionOptions();
                sessionOptions.IntraOpNumThreads = 1;
                sessionOptions.InterOpNumThreads = 1;

                _model = new InferenceSession(modelPath, sessionOptions);
                Console.WriteLine($"Loaded ONNX model from {modelPath}");
            }
            catch (Exception e)
            {
                throw new Exception($"Error loading ONNX model from {modelPath}: {e.Message}", e);
            }

            // Initialize hidden states
            _h01 = new float[1, 1, hiddenSize];
            _h02 = new float[1, 1, hiddenSize];
        }

        private float[] CreateHammingWindow(int length)
        {
            // Equivalent to np.sqrt(np.hamming(length))
            var window = new float[length];
            for (int i = 0; i < length; i++)
            {
                double hamming = 0.54 - 0.46 * Math.Cos(2.0 * Math.PI * i / (length - 1));
                window[i] = (float)Math.Sqrt(hamming);
            }
            return window;
        }

        private static float[] LogPow(float[] signal)
        {
            // Equivalent to np.log10(np.maximum(signal**2, 1e-12))
            var result = new float[signal.Length];
            for (int i = 0; i < signal.Length; i++)
            {
                float pspec = Math.Max(signal[i] * signal[i], 1e-12f);
                result[i] = (float)Math.Log10(pspec);
            }
            return result;
        }

        private static (float[] magnitude, Complex[] phasor) MagPhasor(Complex[] complexSpec)
        {
            // Equivalent to Python's magphasor function
            var magnitude = new float[complexSpec.Length];
            var phasor = new Complex[complexSpec.Length];

            for (int i = 0; i < complexSpec.Length; i++)
            {
                magnitude[i] = (float)complexSpec[i].Magnitude;
                if (magnitude[i] == 0.0f)
                {
                    phasor[i] = Complex.One;
                }
                else
                {
                    phasor[i] = complexSpec[i] / magnitude[i];
                }
            }

            return (magnitude, phasor);
        }

        private float[,,] CalcFeatures(float[] xmagMic, float[] xmagFar)
        {
            var featMic = LogPow(xmagMic);
            var featFar = LogPow(xmagFar);

            // Equivalent to np.concatenate([feat_mic, feat_far])
            var feat = new float[1, 1, featMic.Length + featFar.Length];

            for (int i = 0; i < featMic.Length; i++)
            {
                feat[0, 0, i] = featMic[i] / 20.0f;
            }

            for (int i = 0; i < featFar.Length; i++)
            {
                feat[0, 0, featMic.Length + i] = featFar[i] / 20.0f;
            }

            return feat;
        }

        private Complex[] RFFT(float[] input, int nfft)
        {
            // Equivalent to np.fft.rfft(input, nfft)
            // Create padded input array
            var paddedInput = new double[nfft];
            for (int i = 0; i < Math.Min(input.Length, nfft); i++)
            {
                paddedInput[i] = input[i];
            }

            // Convert to complex array for FFT
            var complexInput = paddedInput.Select(x => new Complex(x, 0)).ToArray();

            // Perform FFT using Math.NET
            Fourier.Forward(complexInput, FourierOptions.Matlab);

            // Return only the positive frequencies (like rfft)
            var result = new Complex[nfft / 2 + 1];
            Array.Copy(complexInput, result, result.Length);

            return result;
        }

        private float[] IRFFT(Complex[] input, int nfft)
        {
            // Equivalent to np.fft.irfft(input, nfft)
            // Reconstruct full complex spectrum from rfft output
            var fullSpectrum = new Complex[nfft];

            // Copy positive frequencies
            Array.Copy(input, fullSpectrum, input.Length);

            // Mirror negative frequencies (conjugate symmetry)
            for (int i = 1; i < input.Length - 1; i++)
            {
                fullSpectrum[nfft - i] = Complex.Conjugate(input[i]);
            }

            // Perform inverse FFT
            Fourier.Inverse(fullSpectrum, FourierOptions.Matlab);

            // Extract real part and convert to float
            var result = new float[nfft];
            for (int i = 0; i < nfft; i++)
            {
                result[i] = (float)fullSpectrum[i].Real;
            }

            return result;
        }

        private float[] FlattenArray(float[,,] array)
        {
            // Equivalent to numpy array flattening
            var flat = new float[array.GetLength(0) * array.GetLength(1) * array.GetLength(2)];
            int index = 0;
            for (int i = 0; i < array.GetLength(0); i++)
                for (int j = 0; j < array.GetLength(1); j++)
                    for (int k = 0; k < array.GetLength(2); k++)
                        flat[index++] = array[i, j, k];
            return flat;
        }

        private void UpdateHiddenState(float[,,] target, float[] source)
        {
            // Update 3D array from flat array
            int index = 0;
            for (int i = 0; i < target.GetLength(0); i++)
                for (int j = 0; j < target.GetLength(1); j++)
                    for (int k = 0; k < target.GetLength(2); k++)
                        target[i, j, k] = source[index++];
        }

        private float[] ElementwiseMultiply(float[] a, float[] b)
        {
            // Equivalent to numpy element-wise multiplication
            var result = new float[Math.Min(a.Length, b.Length)];
            for (int i = 0; i < result.Length; i++)
            {
                result[i] = a[i] * b[i];
            }
            return result;
        }

        public float[] EnhanceFrame(float[] micFrame, float[] farFrame)
        {
            // Ensure frames are the correct size
            var processedMicFrame = new float[_frameSize];
            var processedFarFrame = new float[_frameSize];

            Array.Copy(micFrame, processedMicFrame, Math.Min(micFrame.Length, _frameSize));
            Array.Copy(farFrame, processedFarFrame, Math.Min(farFrame.Length, _frameSize));

            // Apply window to input frames (equivalent to mic_frame * window)
            var windowedMic = ElementwiseMultiply(processedMicFrame, _window);
            var windowedFar = ElementwiseMultiply(processedFarFrame, _window);

            // FFT (equivalent to np.fft.rfft(windowed_frame, dft_size))
            var cspecMic = RFFT(windowedMic, _dftSize);
            var (xmagMic, xphsMic) = MagPhasor(cspecMic);

            var cspecFar = RFFT(windowedFar, _dftSize);
            var xmagFar = cspecFar.Select(c => (float)c.Magnitude).ToArray();

            // Calculate features
            var feat = CalcFeatures(xmagMic, xmagFar);

            // Prepare inputs for ONNX model with flattened arrays
            var inputTensor = new DenseTensor<float>(FlattenArray(feat), new[] { 1, 1, feat.GetLength(2) });
            var h01Tensor = new DenseTensor<float>(FlattenArray(_h01), new[] { 1, 1, _hiddenSize });
            var h02Tensor = new DenseTensor<float>(FlattenArray(_h02), new[] { 1, 1, _hiddenSize });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input", inputTensor),
                NamedOnnxValue.CreateFromTensor("h01", h01Tensor),
                NamedOnnxValue.CreateFromTensor("h02", h02Tensor)
            };

            // Run inference
            using var results = _model.Run(inputs);
            var outputTensors = results.ToArray();

            // Extract outputs
            var maskTensor = outputTensors[0].AsTensor<float>();
            var newH01Tensor = outputTensors[1].AsTensor<float>();
            var newH02Tensor = outputTensors[2].AsTensor<float>();

            // Update hidden states from flat arrays
            UpdateHiddenState(_h01, newH01Tensor.ToArray());
            UpdateHiddenState(_h02, newH02Tensor.ToArray());

            // Apply mask and reconstruct (equivalent to mask * xmag_mic * xphs_mic)
            var mask = maskTensor.ToArray();
            var enhancedSpectrum = new Complex[cspecMic.Length];

            for (int i = 0; i < cspecMic.Length; i++)
            {
                enhancedSpectrum[i] = mask[i] * xmagMic[i] * xphsMic[i];
            }

            // IFFT (equivalent to np.fft.irfft(enhanced_spectrum, dft_size))
            var enhancedFrame = IRFFT(enhancedSpectrum, _dftSize);

            // Apply window again and trim to frame size
            var windowedEnhanced = ElementwiseMultiply(enhancedFrame.Take(_frameSize).ToArray(), _window);

            return windowedEnhanced;
        }

        public void Dispose()
        {
            _model?.Dispose();
        }
    }

    public class EchoCancellationAudioProcessor : IDisposable
    {
        private readonly DECModelRealTime _echoCanceller;
        private readonly WaveInEvent _waveIn;
        private readonly WaveOutEvent _waveOut;
        private readonly BufferedWaveProvider _outputBuffer;
        private readonly WaveFileWriter _micWaveWriter;
        private readonly WaveFileWriter _refWaveWriter;
        private readonly WaveFileWriter _ecWaveWriter;

        private readonly int _frameSize;
        private readonly int _sampleRate;
        private readonly WaveFormat _waveFormat;

        private readonly ConcurrentQueue<float[]> _micFrameQueue;
        private readonly ConcurrentQueue<float[]> _referenceFrameQueue;
        private readonly object _lockObject = new object();

        private float[] _micBuffer;
        private float[] _referenceBuffer;
        private int _micBufferPosition;
        private int _refBufferPosition;

        private bool _disposed = false;
        private volatile bool _isProcessing = false;

        public EchoCancellationAudioProcessor(string modelPath, int frameSize = 320, int sampleRate = 16000,
            int micDeviceNumber = -1, int speakerDeviceNumber = -1, bool saveToWav = true)
        {
            _frameSize = frameSize;
            _sampleRate = sampleRate;
            _waveFormat = new WaveFormat(sampleRate, 16, 1);

            // Initialize buffers
            _micFrameQueue = new ConcurrentQueue<float[]>();
            _referenceFrameQueue = new ConcurrentQueue<float[]>();
            _micBuffer = new float[frameSize];
            _referenceBuffer = new float[frameSize];

            // Initialize echo canceller
            _echoCanceller = new DECModelRealTime(
                modelPath: modelPath,
                windowLength: (double)frameSize / sampleRate,
                hopFraction: 0.0f,
                dftSize: 320,
                hiddenSize: 322,
                samplingRate: sampleRate
            );

            Console.WriteLine("Echo canceller initialized.");

            // Initialize audio input (microphone)
            _waveIn = new WaveInEvent
            {
                WaveFormat = _waveFormat,
                BufferMilliseconds = 20 // 20ms buffer for low latency
            };
            _waveIn.DataAvailable += OnMicrophoneDataAvailable;

            // Initialize audio output (speaker)
            _outputBuffer = new BufferedWaveProvider(_waveFormat)
            {
                BufferDuration = TimeSpan.FromMilliseconds(500),
                DiscardOnBufferOverflow = true
            };

            _waveOut = new WaveOutEvent
            {
                DesiredLatency = 40 // Low latency
            };
            _waveOut.Init(_outputBuffer);

            // Initialize WAV file writers if requested
            if (saveToWav)
            {
                _micWaveWriter = new WaveFileWriter("mic.wav", _waveFormat);
                _refWaveWriter = new WaveFileWriter("ref.wav", _waveFormat);
                _ecWaveWriter = new WaveFileWriter("ec.wav", _waveFormat);

                Console.WriteLine("Outputting to WAV files: mic.wav, ref.wav, ec.wav");
            }
        }

        private void OnMicrophoneDataAvailable(object sender, WaveInEventArgs e)
        {
            if (!_isProcessing) return;

            Console.WriteLine($"Microphone data received: {e.BytesRecorded} bytes");

            // Convert byte array to float array (equivalent to converting to numpy array)
            var samples = new float[e.BytesRecorded / 2]; // 16-bit = 2 bytes per sample
            for (int i = 0; i < samples.Length; i++)
            {
                short sample = BitConverter.ToInt16(e.Buffer, i * 2);
                samples[i] = sample / 32768f; // Convert to float [-1, 1]
            }

            lock (_lockObject)
            {
                // Add samples to microphone buffer
                for (int i = 0; i < samples.Length; i++)
                {
                    _micBuffer[_micBufferPosition] = samples[i];
                    _micBufferPosition++;

                    if (_micBufferPosition >= _frameSize)
                    {
                        // Frame is complete, queue it for processing
                        var frame = new float[_frameSize];
                        Array.Copy(_micBuffer, frame, _frameSize);
                        _micFrameQueue.Enqueue(frame);
                        _micBufferPosition = 0;
                    }
                }

                // For demonstration, we'll use the microphone input as reference too
                // In a real application, you'd capture the speaker output separately
                for (int i = 0; i < samples.Length; i++)
                {
                    _referenceBuffer[_refBufferPosition] = samples[i] * 0.5f; // Simulate speaker reference
                    _refBufferPosition++;

                    if (_refBufferPosition >= _frameSize)
                    {
                        var frame = new float[_frameSize];
                        Array.Copy(_referenceBuffer, frame, _frameSize);
                        _referenceFrameQueue.Enqueue(frame);
                        _refBufferPosition = 0;
                    }
                }

                // Process frames if both queues have data
                ProcessQueuedFrames();
            }

            // Save microphone input to WAV file
            _micWaveWriter?.Write(e.Buffer, 0, e.BytesRecorded);
        }

        private void ProcessQueuedFrames()
        {
            while (_micFrameQueue.TryDequeue(out var micFrame) &&
                   _referenceFrameQueue.TryDequeue(out var refFrame))
            {
                try
                {
                    // Echo cancellation processing
                    var processedData = _echoCanceller.EnhanceFrame(micFrame, refFrame);
                    //processedData = DynamicRangeCompression(processedData, threshold: 0.5f, ratio: 4.0f);

                    // Convert processed audio back to bytes for output
                    var outputBytes = ConvertFloatToBytes(processedData);
                    if (outputBytes.Length == 0)
                    {
                        Console.WriteLine("Processed data is empty, skipping output.");
                        continue;
                    }
                    // Add to output buffer for speaker playback
                    _outputBuffer.AddSamples(outputBytes, 0, outputBytes.Length);

                    // Save to WAV files
                    var refBytes = ConvertFloatToBytes(refFrame);
                    var ecBytes = ConvertFloatToBytes(processedData);

                    _refWaveWriter?.Write(refBytes, 0, refBytes.Length);
                    _ecWaveWriter?.Write(ecBytes, 0, ecBytes.Length);
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"Error processing audio frame: {ex.Message}");
                }
            }
        }

        private float[] DynamicRangeCompression(float[] signal, float threshold = 0.5f, float ratio = 4.0f)
        {
            // Equivalent to numpy where operation for dynamic range compression
            var result = new float[signal.Length];
            for (int i = 0; i < signal.Length; i++)
            {
                float absValue = Math.Abs(signal[i]);
                if (absValue > threshold)
                {
                    result[i] = Math.Sign(signal[i]) * (threshold + (absValue - threshold) / ratio);
                }
                else
                {
                    result[i] = signal[i];
                }
            }
            return result;
        }

        private byte[] ConvertFloatToBytes(float[] samples)
        {
            var bytes = new byte[samples.Length * 2];
            for (int i = 0; i < samples.Length; i++)
            {
                short sample = (short)(Math.Max(-1.0f, Math.Min(1.0f, samples[i])) * short.MaxValue);
                byte[] sampleBytes = BitConverter.GetBytes(sample);
                bytes[i * 2] = sampleBytes[0];
                bytes[i * 2 + 1] = sampleBytes[1];
            }
            return bytes;
        }

        public void StartProcessing()
        {
            if (_isProcessing) return;

            _isProcessing = true;
            _waveOut.Play();
            _waveIn.StartRecording();

            Console.WriteLine("Audio processing started. Press any key to stop...");
        }

        public void StopProcessing()
        {
            if (!_isProcessing) return;

            _isProcessing = false;
            _waveIn?.StopRecording();
            _waveOut?.Stop();

            Console.WriteLine("Audio processing stopped.");
        }

        public void Dispose()
        {
            if (!_disposed)
            {
                StopProcessing();

                _waveIn?.Dispose();
                _waveOut?.Dispose();
                _outputBuffer?.ClearBuffer();
                _echoCanceller?.Dispose();
                _micWaveWriter?.Dispose();
                _refWaveWriter?.Dispose();
                _ecWaveWriter?.Dispose();

                _disposed = true;
            }
        }
    }

    class Program
    {
        static void Main(string[] args)
        {
            const string modelPath = "dec-baseline-model-icassp2022.onnx";

            // List available audio devices
            Console.WriteLine("=== Available Audio Input Devices ===");


            Console.WriteLine("\n=== Available Audio Output Devices ===");


            Console.WriteLine("\nUsage: Press Enter to use default devices, or specify device numbers:");
            Console.Write("Microphone device number (default 0): ");
            var micInput = Console.ReadLine();
            int micDevice = string.IsNullOrEmpty(micInput) ? 0 : int.Parse(micInput);

            Console.Write("Speaker device number (default 0): ");
            var speakerInput = Console.ReadLine();
            int speakerDevice = string.IsNullOrEmpty(speakerInput) ? 0 : int.Parse(speakerInput);

            try
            {
                using var processor = new EchoCancellationAudioProcessor(
                    modelPath,
                    frameSize: 320,
                    sampleRate: 16000,
                    micDeviceNumber: micDevice,
                    speakerDeviceNumber: speakerDevice,
                    saveToWav: true);

                processor.StartProcessing();
                Console.ReadKey();
                processor.StopProcessing();

                Console.WriteLine("Audio processing terminated.");
            }
            catch (Exception ex)
            {
                Console.WriteLine($"Error: {ex.Message}");
                Console.WriteLine("Make sure the ONNX model file exists and audio devices are available.");
            }
        }
    }
}