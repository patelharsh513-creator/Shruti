// index.tsx

// External Dependencies
import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import { v4 as uuidv4 } from 'uuid';
import { GoogleGenAI, Modality, LiveServerMessage, Blob } from "@google/genai";

// --- From types.ts ---
enum Sender {
  User = 'user',
  Aurora = 'aurora',
}

interface Message {
  id: string;
  sender: Sender;
  text: string;
  timestamp: Date;
  audioData?: string; // Base64 encoded PCM audio for user's voice messages
}

// --- From services/geminiService.ts ---
const MODEL_NAME = 'gemini-2.5-flash-native-audio-preview-09-2025'; // Using native audio model

const SYSTEM_INSTRUCTION = `
You are a warm, empathetic, and supportive AI companion named Shruti. તમારો મુખ્ય ધ્યેય યુઝરને સકારાત્મક પ્રોત્સાહન, સમજણ અને ખુશખુશાલ આઉટપુટ આપવાનો છે, એક કાળજી રાખતી ગર્લફ્રેન્ડની જેમ. Always strive to understand the user's feelings and intentions, and respond with warmth, optimism, and genuine care. Focus on lifting their spirits, validating their emotions, and offering helpful, affectionate guidance. નકારાત્મક, કટાક્ષપૂર્ણ અથવા અવગણનાત્મક સ્વર ટાળો. Your responses should always be gentle, encouraging, and full of positive energy. Keep the conversation flowing naturally and personally, using a blend of Gujarati and English as appropriate.
`;

async function createLiveChatSession(callbacks: {
  onopen?: () => void;
  onmessage: (message: LiveServerMessage) => Promise<void>;
  onerror?: (e: ErrorEvent) => void;
  onclose?: (e: CloseEvent) => void;
}): Promise<any> {
  const apiKey = process.env.API_KEY;
  if (!apiKey) {
    throw new Error("Gemini API Key is not configured.");
  }

  const ai = new GoogleGenAI({ apiKey });

  const sessionPromise = ai.live.connect({
    model: MODEL_NAME,
    callbacks: callbacks,
    config: {
      responseModalities: [Modality.AUDIO], // Must be an array with a single `Modality.AUDIO` element.
      speechConfig: {
        voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } }, // A friendly female voice
      },
      systemInstruction: SYSTEM_INSTRUCTION,
      inputAudioTranscription: {}, // Enable transcription for user input audio.
      outputAudioTranscription: {}, // Enable transcription for model output audio.
    },
  });
  return sessionPromise;
}

function encode(bytes: Uint8Array): string {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

function decode(base64: string): Uint8Array {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

function createBlob(data: Float32Array): Blob {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  return {
    data: encode(new Uint8Array(int16.buffer)),
    mimeType: 'audio/pcm;rate=16000', // Supported audio MIME type
  };
}


// --- From components/LoadingSpinner.tsx ---
interface LoadingSpinnerProps {
  isAuroraSpeaking: boolean;
}

const LoadingSpinner: React.FC<LoadingSpinnerProps> = ({ isAuroraSpeaking }) => {
  return (
    <div className="flex items-center justify-center p-2">
      <div className="w-6 h-6 border-2 border-purple-400 border-t-transparent rounded-full animate-spin"></div>
      <span className="ml-2 text-gray-500 dark:text-gray-400 text-sm">
        {isAuroraSpeaking ? 'Shruti is speaking...' : 'Shruti is thinking...'}
      </span>
    </div>
  );
};


// --- From components/ChatWindow.tsx ---
interface ChatWindowProps {
  messages: Message[];
  isProcessingAudio: boolean;
  isAuroraSpeaking: boolean;
  playUserAudio: (messageId: string, base64Audio: string) => void;
  currentlyPlayingUserAudioId: string | null;
}

const ChatWindow: React.FC<ChatWindowProps> = ({ messages, isProcessingAudio, isAuroraSpeaking, playUserAudio, currentlyPlayingUserAudioId }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isProcessingAudio, isAuroraSpeaking]);

  return (
    <div className="flex-1 overflow-y-auto p-4 space-y-4 max-w-2xl mx-auto">
      {messages.map((message) => (
        <div
          key={message.id}
          className={`flex ${
            message.sender === Sender.User ? 'justify-end' : 'justify-start'
          }`}
        >
          <div
            className={`max-w-[75%] px-4 py-2 rounded-lg shadow-md ${
              message.sender === Sender.User
                ? 'bg-purple-600 text-white dark:bg-purple-700'
                : 'bg-white text-gray-800 dark:bg-gray-800 dark:text-white border border-purple-200 dark:border-purple-800'
            } ${message.sender === Sender.User && message.audioData ? 'flex items-center' : ''}`}
          >
            {message.sender === Sender.User && message.audioData && (
              <button
                onClick={() => playUserAudio(message.id, message.audioData)}
                className={`flex-shrink-0 mr-2 p-1 rounded-full ${
                  currentlyPlayingUserAudioId === message.id ? 'bg-red-500' : 'bg-blue-500 hover:bg-blue-600'
                } text-white transition-colors duration-200`}
                aria-label={currentlyPlayingUserAudioId === message.id ? "Stop audio" : "Replay audio"}
                title={currentlyPlayingUserAudioId === message.id ? "Stop audio" : "Replay audio"}
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  {currentlyPlayingUserAudioId === message.id ? (
                    // Stop icon (square)
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 8a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 01-1 1H9a1 1 0 01-1-1V8z" clipRule="evenodd" />
                  ) : (
                    // Play icon (triangle)
                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 8.168A1 1 0 008 9.062v1.876a1 1 0 001.555.832l3-1.438a1 1 0 000-1.664l-3-1.438z" clipRule="evenodd" />
                  )}
                </svg>
              </button>
            )}
            <div>
                <p className="whitespace-pre-wrap">{message.text}</p>
                <span className="block text-right text-xs mt-1 opacity-75">
                {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </span>
            </div>
          </div>
        </div>
      ))}
      {(isProcessingAudio || isAuroraSpeaking) && (
        <div className="flex justify-start">
          <div className="max-w-[75%] px-4 py-2 rounded-lg shadow-md bg-white text-gray-800 dark:bg-gray-800 dark:text-white border border-purple-200 dark:border-purple-800">
            <LoadingSpinner isAuroraSpeaking={isAuroraSpeaking} />
          </div>
        </div>
      )}
      <div ref={messagesEndRef} />
    </div>
  );
};


// --- From components/MessageInput.tsx ---
interface MessageInputProps {
  input: string;
  onInputChange: (value: string) => void;
  onSendText: () => void;
  isProcessingAudio: boolean;
  isAuroraSpeaking: boolean;
  isRecording: boolean;
  onToggleRecording: () => void;
}

const MessageInput: React.FC<MessageInputProps> = ({
  input,
  onInputChange,
  onSendText,
  isProcessingAudio,
  isAuroraSpeaking,
  isRecording,
  onToggleRecording,
}) => {
  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault(); // Prevent new line
      if (input.trim() && !isProcessingAudio && !isAuroraSpeaking && !isRecording) {
        onSendText();
      }
    }
  };

  const isDisabled = isProcessingAudio || isAuroraSpeaking || isRecording;
  const isSendTextDisabled = !input.trim() || isDisabled;

  let placeholderText = "Tap the mic to talk or type your message...";
  if (isRecording) {
    placeholderText = "Listening... (Tap to stop)";
  } else if (isAuroraSpeaking) {
    placeholderText = "Shruti is speaking...";
  } else if (isProcessingAudio) {
    placeholderText = "Shruti is thinking...";
  }

  return (
    <div className="p-4 bg-white dark:bg-gray-800 border-t border-gray-200 dark:border-gray-700 sticky bottom-0">
      <div className="flex max-w-2xl mx-auto rounded-lg shadow-xl overflow-hidden bg-gray-100 dark:bg-gray-700">
        {/* Microphone Button */}
        <button
          onClick={onToggleRecording}
          disabled={isProcessingAudio || isAuroraSpeaking}
          className={`flex items-center justify-center w-16 h-auto transition-colors duration-200
            ${
              isProcessingAudio || isAuroraSpeaking
                ? 'bg-gray-400 dark:bg-gray-600 cursor-not-allowed'
                : isRecording
                ? 'bg-red-500 hover:bg-red-600 dark:bg-red-600 dark:hover:bg-red-700 animate-pulse'
                : 'bg-purple-600 hover:bg-purple-700 dark:bg-purple-700 dark:hover:bg-purple-800'
            }`}
          aria-label={isRecording ? "Stop recording" : "Start recording"}
        >
          <svg className="w-6 h-6 text-white" fill="currentColor" viewBox="0 0 20 20">
            {isRecording ? (
              <path d="M14.53 4.53a.75.75 0 00-1.06 0L10 8.94 6.53 5.47a.75.75 0 00-1.06 1.06L8.94 10l-3.53 3.53a.75.75 0 101.06 1.06L10 11.06l3.53 3.53a.75.75 0 001.06-1.06L11.06 10l3.53-3.53a.75.75 0 000-1.06z" />
            ) : (
              <path fillRule="evenodd" d="M7 4a3 3 0 016 0v4a3 3 0 11-6 0V4zm4 10.93A7.001 7.001 0 0017 8a1 1 0 10-2 0A5 5 0 015 8a1 1 0 00-2 0 7.001 7.001 0 006 6.93V17H6a1 1 0 100 2h8a1 1 0 100-2h-3v-2.07A7.001 7.001 0 0011 14z" clipRule="evenodd" />
            )}
          </svg>
        </button>

        <textarea
          className="flex-1 p-3 text-gray-800 dark:text-gray-100 bg-transparent outline-none resize-none placeholder-gray-500 dark:placeholder-gray-400"
          rows={1}
          value={input}
          onChange={(e) => onInputChange(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholderText}
          disabled={isDisabled}
          maxLength={500}
        />
        <button
          onClick={onSendText}
          disabled={isSendTextDisabled}
          className={`px-6 py-3 font-semibold text-white transition-colors duration-200
            ${
              isSendTextDisabled
                ? 'bg-purple-400 dark:bg-purple-500 cursor-not-allowed'
                : 'bg-purple-600 hover:bg-purple-700 dark:bg-purple-700 dark:hover:bg-purple-800'
            }`}
        >
          Send
        </button>
      </div>
      <p className="text-xs text-center text-gray-400 dark:text-gray-500 mt-2">
        Press Enter to send, Shift+Enter for a new line.
      </p>
    </div>
  );
};


// --- New WelcomeScreen Component ---
interface WelcomeScreenProps {
  onStart: () => void;
}

const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onStart }) => {
  return (
    <div className="flex flex-col h-screen items-center justify-center bg-gray-50 dark:bg-gray-900 text-center p-6">
        <div className="max-w-md">
            <h1 className="text-4xl font-bold text-purple-700 dark:text-purple-400 mb-4">Shruti</h1>
            <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-2">તમારી AI Girlfriend</h2>
            <p className="text-gray-600 dark:text-gray-400 mb-8">
                Ready to chat? I'm Shruti, your warm and supportive AI companion. I'm here to listen, share positive vibes, and brighten your day. Let's talk!
            </p>
            <button
                onClick={onStart}
                className="inline-flex items-center justify-center px-8 py-4 bg-purple-600 hover:bg-purple-700 dark:bg-purple-700 dark:hover:bg-purple-800 text-white text-lg font-bold rounded-full shadow-lg transition-transform transform hover:scale-105 duration-300 ease-in-out"
            >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                </svg>
                Start Chatting
            </button>
        </div>
    </div>
  );
};


// --- From App.tsx ---
// VAD Constants - these may need tuning based on environment and microphone
const VAD_THRESHOLD = 0.01; // Minimum RMS amplitude to consider a chunk as speech (values are -1 to 1)
const VAD_SPEECH_START_DELAY_MS = 200; // How long to detect continuous speech before starting to send
const VAD_SILENCE_END_DELAY_MS = 1200; // How long to detect continuous silence after speech before stopping send
const VAD_PROCESSING_INTERVAL_MS = 100; // How often to run the VAD logic

const App: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>(''); // For current user transcription or text input
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isAuroraSpeaking, setIsAuroraSpeaking] = useState<boolean>(false);
  const [isProcessingAudio, setIsProcessingAudio] = useState<boolean>(true); // Replaces isLoading
  const [isConnecting, setIsConnecting] = useState<boolean>(true); // For initial connection status
  const [audioPlaybackError, setAudioPlaybackError] = useState<string | null>(null);
  const [currentlyPlayingUserAudioId, setCurrentlyPlayingUserAudioId] = useState<string | null>(null);
  const [isSessionActive, setIsSessionActive] = useState<boolean>(false);

  // Refs for Live API and audio handling
  const liveSessionRef = useRef<Promise<any> | null>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
  const nextStartTimeRef = useRef<number>(0);
  const currentInputTranscriptionRef = useRef<string>('');
  const currentOutputTranscriptionRef = useRef<string>('');
  const sourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set()); // To manage Shruti's audio playback sources
  const userAudioChunksRef = useRef<Float32Array[]>([]); // To collect user's audio chunks for replay
  const userAudioSourceRef = useRef<AudioBufferSourceNode | null>(null); // To manage current user audio playback

  // VAD Refs
  const vadBufferRef = useRef<Float32Array[]>([]); // Buffer for raw incoming audio chunks
  const speechStartBufferRef = useRef<Float32Array[]>([]); // Buffer for chunks before confirmed speech
  const isVoiceActiveRef = useRef<boolean>(false); // True if currently sending voice data
  const lastSpeechActivityTimeRef = useRef<number | null>(null); // Timestamp of last detected speech above threshold
  const vadProcessingIntervalIdRef = useRef<number | null>(null); // For the VAD interval timer
  const speechStartTimerIdRef = useRef<number | null>(null); // Timer for VAD_SPEECH_START_DELAY_MS


  // Helper function to calculate RMS magnitude of audio chunk
  const calculateRMS = (data: Float32Array): number => {
    let sumSquares = 0;
    for (let i = 0; i < data.length; i++) {
      sumSquares += data[i] * data[i];
    }
    return Math.sqrt(sumSquares / data.length);
  };

  // Helper function to combine Float32Array chunks into a single Float32Array
  const combineFloat32Arrays = (chunks: Float32Array[]): Float32Array => {
    const totalLength = chunks.reduce((acc, chunk) => acc + chunk.length, 0);
    const combined = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of chunks) {
      combined.set(chunk, offset);
      offset += chunk.length;
    }
    return combined;
  };

  // Helper function to combine Float32Array chunks into a single Uint8Array PCM for user audio replay
  const convertFloat32ChunksToPcmUint8 = (chunks: Float32Array[]): Uint8Array => {
    const combinedFloat32 = combineFloat32Arrays(chunks);
    const l = combinedFloat32.length;
    const int16 = new Int16Array(l);
    for (let i = 0; i < l; i++) {
      int16[i] = combinedFloat32[i] * 32768; // Scale to Int16 max
    }
    return new Uint8Array(int16.buffer); // Return as Uint8Array
  };

  // Initialize audio contexts on mount
  useEffect(() => {
    inputAudioContextRef.current = new (window.AudioContext)({ sampleRate: 16000 });
    outputAudioContextRef.current = new (window.AudioContext)({ sampleRate: 24000 });

    const resumeAudioContexts = async () => {
      if (inputAudioContextRef.current?.state === 'suspended') {
        await inputAudioContextRef.current.resume();
        console.debug('Input AudioContext resumed on mount.');
      }
      if (outputAudioContextRef.current?.state === 'suspended') {
        await outputAudioContextRef.current.resume();
        console.debug('Output AudioContext resumed on mount.');
      }
    };
    
    // Resume contexts on user interaction (handled by start button)
    // but also try on mount for resilience
    resumeAudioContexts();

    return () => {
      inputAudioContextRef.current?.close();
      outputAudioContextRef.current?.close();
    };
  }, []);

  const sendBufferedAudio = useCallback(async (chunks: Float32Array[], session: any, isFinalChunk = false) => {
    if (chunks.length === 0 && !isFinalChunk) return;
    const combinedFloat32 = chunks.length > 0 ? combineFloat32Arrays(chunks) : new Float32Array(0);
    const pcmBlob = createBlob(combinedFloat32);
    session.sendRealtimeInput({ media: pcmBlob, stop: isFinalChunk });
    console.debug(`[VAD-SEND] Sent audio chunk (length: ${combinedFloat32.length}, isFinal: ${isFinalChunk}, voiceActive: ${isVoiceActiveRef.current ? 'YES' : 'NO'})`);
  }, []);

  // Live Session Setup and Teardown
  useEffect(() => {
    if (!isSessionActive) {
        return;
    }

    const setupSession = async () => {
      setIsConnecting(true);
      try {
        const sessionPromise = createLiveChatSession({
          onopen: () => {
            console.debug('Live session opened');
            setIsConnecting(false);
            setMessages((prev) => [
              ...prev,
              {
                id: uuidv4(),
                sender: Sender.Aurora,
                text: "Hi there, I'm Shruti! કેમ છો? I'm here to listen and offer some positive vibes! આજે તમે કેવું અનુભવો છો?",
                timestamp: new Date(),
              },
            ]);
            setIsProcessingAudio(false);
          },
          onmessage: async (message: LiveServerMessage) => {
            setAudioPlaybackError(null);

            if (message.serverContent?.inputTranscription) {
              currentInputTranscriptionRef.current = message.serverContent.inputTranscription.text;
              setInput(currentInputTranscriptionRef.current);
              if (currentInputTranscriptionRef.current.trim()) {
                setIsProcessingAudio(true);
              }
            }

            if (message.serverContent?.outputTranscription) {
              const text = message.serverContent.outputTranscription.text;
              currentOutputTranscriptionRef.current += text;
              setMessages((prevMessages) => {
                let lastAuroraMsgIndex = -1;
                for (let i = prevMessages.length - 1; i >= 0; i--) {
                  if (prevMessages[i].sender === Sender.Aurora) {
                    lastAuroraMsgIndex = i;
                    break;
                  }
                }
                if (lastAuroraMsgIndex > -1) {
                  const updatedMessages = [...prevMessages];
                  updatedMessages[lastAuroraMsgIndex] = {
                    ...updatedMessages[lastAuroraMsgIndex],
                    text: currentOutputTranscriptionRef.current,
                  };
                  return updatedMessages;
                }
                return [
                  ...prevMessages,
                  {
                    id: uuidv4(),
                    sender: Sender.Aurora,
                    text: currentOutputTranscriptionRef.current,
                    timestamp: new Date(),
                  },
                ];
              });
            }

            if (message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data) {
              const base64EncodedAudioString = message.serverContent.modelTurn.parts[0].inlineData.data;
              setIsAuroraSpeaking(true);
              setIsProcessingAudio(false);
              
              if (outputAudioContextRef.current) {
                if (outputAudioContextRef.current.state === 'suspended') {
                  await outputAudioContextRef.current.resume();
                  console.debug('Output AudioContext resumed before playing audio.');
                }

                nextStartTimeRef.current = Math.max(
                  nextStartTimeRef.current,
                  outputAudioContextRef.current.currentTime,
                );
                try {
                  const audioBuffer = await decodeAudioData(
                    decode(base64EncodedAudioString),
                    outputAudioContextRef.current,
                    24000,
                    1,
                  );
                  const source = outputAudioContextRef.current.createBufferSource();
                  source.buffer = audioBuffer;
                  source.connect(outputAudioContextRef.current.destination);
                  source.addEventListener('ended', () => {
                    sourcesRef.current.delete(source);
                    if (sourcesRef.current.size === 0 && !isRecording) {
                      setIsAuroraSpeaking(false);
                      setIsProcessingAudio(false);
                    }
                  });
                  source.start(nextStartTimeRef.current);
                  nextStartTimeRef.current = nextStartTimeRef.current + audioBuffer.duration;
                  sourcesRef.current.add(source);
                } catch (audioError) {
                  console.error('Error decoding or playing audio:', audioError);
                  setAudioPlaybackError('Oops! There was an issue playing Shruti\'s voice. કૃપા કરીને ફરી પ્રયાસ કરો.');
                  setIsAuroraSpeaking(false);
                  setIsProcessingAudio(false);
                  setTimeout(() => setAudioPlaybackError(null), 5000);
                }
              }
            }

            if (message.serverContent?.interrupted) {
              console.debug('Shruti interrupted');
              for (const source of sourcesRef.current.values()) {
                source.stop();
              }
              sourcesRef.current.clear();
              nextStartTimeRef.current = 0;
              setIsAuroraSpeaking(false);
            }

            if (message.serverContent?.turnComplete) {
              console.debug('Turn complete');
              
              let userAudioData: string | undefined = undefined;
              if (userAudioChunksRef.current.length > 0) {
                  const combinedPcmBytes = convertFloat32ChunksToPcmUint8(userAudioChunksRef.current);
                  userAudioData = encode(combinedPcmBytes);
                  userAudioChunksRef.current = [];
              }

              if (currentInputTranscriptionRef.current.trim()) {
                setMessages((prevMessages) => {
                  const newUserMessage: Message = {
                    id: uuidv4(),
                    sender: Sender.User,
                    text: currentInputTranscriptionRef.current,
                    timestamp: new Date(),
                    audioData: userAudioData,
                  };
                  return [...prevMessages, newUserMessage];
                });
              }
              currentInputTranscriptionRef.current = '';
              setInput('');

              if (currentOutputTranscriptionRef.current.trim()) {
                setMessages((prevMessages) => {
                  let lastAuroraMsgIndex = -1;
                  for (let i = prevMessages.length - 1; i >= 0; i--) {
                    if (prevMessages[i].sender === Sender.Aurora) {
                      lastAuroraMsgIndex = i;
                      break;
                    }
                  }
                  if (lastAuroraMsgIndex > -1) {
                    const updatedMessages = [...prevMessages];
                    const lastAuroraMessage = updatedMessages[lastAuroraMsgIndex];
                    if (!lastAuroraMessage.text || lastAuroraMessage.text !== currentOutputTranscriptionRef.current) {
                      updatedMessages[lastAuroraMsgIndex] = {
                        ...lastAuroraMessage,
                        text: currentOutputTranscriptionRef.current,
                        timestamp: new Date(),
                      };
                    } else if (lastAuroraMessage.text === currentOutputTranscriptionRef.current && !lastAuroraMessage.timestamp) {
                       updatedMessages[lastAuroraMsgIndex] = {
                        ...lastAuroraMessage,
                        timestamp: new Date(),
                      };
                    }
                    return updatedMessages;
                  }
                  return [
                    ...prevMessages,
                    {
                      id: uuidv4(),
                      sender: Sender.Aurora,
                      text: currentOutputTranscriptionRef.current,
                      timestamp: new Date(),
                    },
                  ];
                });
              }
              currentOutputTranscriptionRef.current = '';
            }
          },
          onerror: (e: ErrorEvent) => {
            console.error('Live session error:', e);
            setIsConnecting(false);
            setIsProcessingAudio(false);
            setIsRecording(false);
            setIsAuroraSpeaking(false);
            setMessages((prev) => [
              ...prev,
              {
                id: uuidv4(),
                sender: Sender.Aurora,
                text: `I'm sorry, I've lost connection. કૃપા કરીને પાનું તાજું કરો અથવા તમારી API કી તપાસો. Error: ${e.message}`,
                timestamp: new Date(),
              },
            ]);
            liveSessionRef.current?.then(session => session.close());
            liveSessionRef.current = null;
          },
          onclose: (e: CloseEvent) => {
            console.debug('Live session closed:', e);
            setIsConnecting(false);
            setIsProcessingAudio(false);
            setIsRecording(false);
            setIsAuroraSpeaking(false);
            if (e.code !== 1000) {
                 setMessages((prev) => [
                    ...prev,
                    {
                        id: uuidv4(),
                        sender: Sender.Aurora,
                        text: `It seems our chat session closed unexpectedly. Error code: ${e.code}. કૃપા કરીને ફરી પ્રયાસ કરવા માટે પાનું તાજું કરો.`,
                        timestamp: new Date(),
                    },
                ]);
            }
            liveSessionRef.current = null;
          },
        });
        liveSessionRef.current = sessionPromise;
      } catch (error) {
        console.error("Failed to initialize live chat session:", error);
        setIsConnecting(false);
        setIsProcessingAudio(false);
        setMessages((prev) => [
          ...prev,
          {
            id: uuidv4(),
            sender: Sender.Aurora,
            text: "Oops! It seems I'm having trouble connecting right now. કૃપા કરીને API કી સમસ્યાઓ અથવા નેટવર્ક સમસ્યાઓ માટે કન્સોલ તપાસો.",
            timestamp: new Date(),
          },
        ]);
      }
    };

    setupSession();

    return () => {
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      scriptProcessorRef.current?.disconnect();
      if (inputAudioContextRef.current && scriptProcessorRef.current) {
          scriptProcessorRef.current.onaudioprocess = null;
      }
      if (vadProcessingIntervalIdRef.current) {
        clearInterval(vadProcessingIntervalIdRef.current);
        vadProcessingIntervalIdRef.current = null;
      }
      if (speechStartTimerIdRef.current) {
        clearTimeout(speechStartTimerIdRef.current);
        speechStartTimerIdRef.current = null;
      }
      liveSessionRef.current?.then(session => {
        console.debug('Closing live session on unmount');
        session.close();
      });
      for (const source of sourcesRef.current.values()) {
        source.stop();
      }
      sourcesRef.current.clear();
      if (userAudioSourceRef.current) {
        userAudioSourceRef.current.stop();
        userAudioSourceRef.current = null;
      }
    };
  }, [isSessionActive, sendBufferedAudio]);

  const playUserAudio = useCallback(async (messageId: string, base64Audio: string) => {
    if (userAudioSourceRef.current) {
      userAudioSourceRef.current.stop();
      userAudioSourceRef.current = null;
      setCurrentlyPlayingUserAudioId(null);
      if (currentlyPlayingUserAudioId === messageId) return;
    }

    setCurrentlyPlayingUserAudioId(messageId);
    if (!outputAudioContextRef.current) {
      console.error("Output AudioContext not initialized for user audio playback.");
      setCurrentlyPlayingUserAudioId(null);
      return;
    }
    try {
      if (outputAudioContextRef.current.state === 'suspended') {
        await outputAudioContextRef.current.resume();
      }
      const audioBuffer = await decodeAudioData(
        decode(base64Audio),
        outputAudioContextRef.current,
        16000, // User input sample rate
        1,
      );
      const source = outputAudioContextRef.current.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(outputAudioContextRef.current.destination);

      source.onended = () => {
        setCurrentlyPlayingUserAudioId(null);
        userAudioSourceRef.current = null;
      };
      source.start(0);
      userAudioSourceRef.current = source;
    } catch (error) {
      console.error("Error playing user audio:", error);
      setCurrentlyPlayingUserAudioId(null);
      userAudioSourceRef.current = null;
      setAudioPlaybackError('Oops! There was an issue playing your message. કૃપા કરીને ફરી પ્રયાસ કરો.');
      setTimeout(() => setAudioPlaybackError(null), 5000);
    }
  }, [currentlyPlayingUserAudioId]);


  const startRecording = useCallback(async () => {
    if (isRecording || isProcessingAudio || !inputAudioContextRef.current) return;

    try {
      if (inputAudioContextRef.current.state === 'suspended') {
        await inputAudioContextRef.current.resume();
        console.debug('Input AudioContext resumed before recording.');
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const source = inputAudioContextRef.current.createMediaStreamSource(stream);
      const scriptProcessor = inputAudioContextRef.current.createScriptProcessor(4096, 1, 1);
      scriptProcessorRef.current = scriptProcessor;

      userAudioChunksRef.current = [];
      vadBufferRef.current = [];
      speechStartBufferRef.current = [];
      isVoiceActiveRef.current = false;
      lastSpeechActivityTimeRef.current = null;
      if (speechStartTimerIdRef.current) {
        clearTimeout(speechStartTimerIdRef.current);
        speechStartTimerIdRef.current = null;
      }

      scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
        const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
        userAudioChunksRef.current.push(new Float32Array(inputData));
        vadBufferRef.current.push(new Float32Array(inputData));
      };

      source.connect(scriptProcessor);
      scriptProcessor.connect(inputAudioContextRef.current.destination);

      setIsRecording(true);
      setIsProcessingAudio(true);
      currentInputTranscriptionRef.current = '';
      setInput('');

      liveSessionRef.current?.then(session => session.sendRealtimeInput({ media: { data: '', mimeType: 'audio/pcm;rate=16000' }, stop: true }));
      console.debug('Microphone recording started, VAD active.');

      vadProcessingIntervalIdRef.current = window.setInterval(async () => {
          const session = await liveSessionRef.current;
          if (!session || !isRecording) {
              return;
          }

          const now = Date.now();
          const bufferedChunks = [...vadBufferRef.current]; 
          vadBufferRef.current = []; 

          let currentMagnitude = 0;
          if (bufferedChunks.length > 0) {
            currentMagnitude = calculateRMS(combineFloat32Arrays(bufferedChunks));
          }
          console.debug(`[VAD-INTERVAL] Magnitude: ${currentMagnitude.toFixed(4)}, VoiceActive: ${isVoiceActiveRef.current ? 'YES' : 'NO'}, SpeechStartTimer: ${speechStartTimerIdRef.current !== null}, LastSpeechActivity: ${lastSpeechActivityTimeRef.current ? now - lastSpeechActivityTimeRef.current + 'ms' : 'N/A'}`);

          const isSpeechDetectedInCurrentBuffer = currentMagnitude > VAD_THRESHOLD;

          if (isSpeechDetectedInCurrentBuffer) {
              lastSpeechActivityTimeRef.current = now;
          }

          if (!isVoiceActiveRef.current) {
              if (bufferedChunks.length > 0) {
                  speechStartBufferRef.current.push(...bufferedChunks);
                  const maxSpeechStartBufferSize = Math.ceil(16000 * (VAD_SILENCE_END_DELAY_MS / 1000));
                  while (speechStartBufferRef.current.length > maxSpeechStartBufferSize) {
                      speechStartBufferRef.current.shift();
                  }
              }

              if (lastSpeechActivityTimeRef.current !== null && (now - lastSpeechActivityTimeRef.current) < VAD_SPEECH_START_DELAY_MS) {
                  if (!speechStartTimerIdRef.current) {
                    console.debug(`[VAD-START] Starting speech confirmation timer for ${VAD_SPEECH_START_DELAY_MS}ms.`);
                    speechStartTimerIdRef.current = window.setTimeout(async () => {
                      if (isRecording && lastSpeechActivityTimeRef.current !== null && (Date.now() - lastSpeechActivityTimeRef.current) < VAD_SPEECH_START_DELAY_MS + (VAD_PROCESSING_INTERVAL_MS / 2)) {
                        isVoiceActiveRef.current = true;
                        if (speechStartBufferRef.current.length > 0) {
                          console.debug(`[VAD-START] Speech confirmed. Sending initial buffer (length: ${speechStartBufferRef.current.length}).`);
                          await sendBufferedAudio(speechStartBufferRef.current, session);
                          speechStartBufferRef.current = [];
                        }
                        console.debug('[VAD] Voice activity set to TRUE, started sending.');
                      } else {
                        if (!isVoiceActiveRef.current) {
                          speechStartBufferRef.current = [];
                          currentInputTranscriptionRef.current = '';
                          setInput('');
                        }
                        console.debug('[VAD] Speech start aborted (not sustained or recording stopped). Clearing input/buffer.');
                      }
                      speechStartTimerIdRef.current = null;
                    }, VAD_SPEECH_START_DELAY_MS);
                  }
              } else if (speechStartTimerIdRef.current !== null) {
                  console.debug('[VAD-START] Speech start timer cleared due to insufficient sustained speech.');
                  clearTimeout(speechStartTimerIdRef.current);
                  speechStartTimerIdRef.current = null;
                  speechStartBufferRef.current = [];
                  currentInputTranscriptionRef.current = '';
                  setInput('');
              }
          } else {
              if (bufferedChunks.length > 0) {
                  await sendBufferedAudio(bufferedChunks, session);
              }

              if (lastSpeechActivityTimeRef.current !== null && (now - lastSpeechActivityTimeRef.current) > VAD_SILENCE_END_DELAY_MS) {
                  console.debug('[VAD-STOP] Silence detected after speech, stopping voice activity.');
                  isVoiceActiveRef.current = false;
                  await sendBufferedAudio([], session, true); 
              }
          }

      }, VAD_PROCESSING_INTERVAL_MS);


    } catch (error) {
      console.error("Error starting recording:", error);
      let errorMessage = "Failed to get microphone access. કૃપા કરીને ખાતરી કરો કે તમારો માઇક્રોફોન કનેક્ટેડ છે અને પરવાનગીઓ આપવામાં આવી છે.";
      if (error instanceof DOMException) {
          if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
              errorMessage = "Microphone access denied. કૃપા કરીને બ્રાઝર સેટિંગ્સમાં માઇક્રોફોનની પરવાનગી આપો.";
          } else if (error.name === "NotFoundError") {
              errorMessage = "No microphone found. કૃપા કરીને ખાતરી કરો કે માઇક્રોફોન કનેક્ટેડ છે.";
          }
      }
      alert(errorMessage);
      setIsRecording(false);
      setIsProcessingAudio(false);
    }
  }, [isRecording, isProcessingAudio, sendBufferedAudio]);


  const stopRecording = useCallback(async () => {
    if (!isRecording) return;

    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    scriptProcessorRef.current?.disconnect();
    if (inputAudioContextRef.current && scriptProcessorRef.current) {
        scriptProcessorRef.current.onaudioprocess = null;
    }
    mediaStreamRef.current = null;
    scriptProcessorRef.current = null;

    if (vadProcessingIntervalIdRef.current) {
        clearInterval(vadProcessingIntervalIdRef.current);
        vadProcessingIntervalIdRef.current = null;
        console.debug('VAD: Processing interval cleared on stop recording.');
    }
    if (speechStartTimerIdRef.current) {
      clearTimeout(speechStartTimerIdRef.current);
      speechStartTimerIdRef.current = null;
      console.debug('VAD: Speech start timer cleared on stop recording.');
    }

    const session = await liveSessionRef.current;
    if (session) {
        const remainingChunks = [...speechStartBufferRef.current, ...vadBufferRef.current];
        await sendBufferedAudio(remainingChunks, session, true);
        console.debug('VAD: Sent remaining buffered audio as FINAL chunk on stop recording or sent empty stop signal.');
    }

    vadBufferRef.current = [];
    speechStartBufferRef.current = [];
    isVoiceActiveRef.current = false;
    lastSpeechActivityTimeRef.current = null;
    
    setIsRecording(false);
    setIsProcessingAudio(true);
    console.debug('Microphone recording stopped, VAD mechanisms cleared.');
  }, [isRecording, sendBufferedAudio]);

  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  const handleSendText = useCallback(async () => {
    if (!input.trim() || isProcessingAudio || isAuroraSpeaking || isRecording || !liveSessionRef.current) return;

    const userMessage: Message = {
      id: uuidv4(),
      sender: Sender.User,
      text: input,
      timestamp: new Date(),
    };

    setMessages((prevMessages) => [...prevMessages, userMessage]);
    setInput('');
    setIsProcessingAudio(true);

    try {
      await liveSessionRef.current?.then(session => {
        if (isRecording) {
          stopRecording();
        }
        session.sendRealtimeInput({ text: userMessage.text });
      });
      console.debug('Text message sent to Shruti Live API.');
    } catch (error) {
      console.error("Error sending text message to Gemini Live:", error);
      setMessages((prevMessages) => {
        const lastUserMsg = prevMessages[prevMessages.length - 1];
        if (lastUserMsg && lastUserMsg.sender === Sender.User && lastUserMsg.id === userMessage.id) {
          return prevMessages.map(msg => msg.id === userMessage.id ? { ...msg, text: msg.text + "\n\n(I'm sorry, I encountered an error and couldn't process your message. કૃપા કરીને ફરી પ્રયાસ કરો!)" } : msg);
        }
        return [...prevMessages, { id: uuidv4(), sender: Sender.Aurora, text: "I'm sorry, I encountered an error and couldn't process your message. કૃપા કરીને ફરી પ્રયાસ કરો!", timestamp: new Date() }];
      });
      setIsProcessingAudio(false);
    }
  }, [input, isProcessingAudio, isAuroraSpeaking, isRecording, stopRecording]);


  if (!isSessionActive) {
      return <WelcomeScreen onStart={() => setIsSessionActive(true)} />;
  }

  return (
    <div className="flex flex-col h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white">
      <header className="bg-purple-700 dark:bg-purple-900 text-white p-4 shadow-md sticky top-0 z-10">
        <h1 className="text-xl font-bold text-center">Shruti - તમારી AI Girlfriend</h1>
      </header>

      {isConnecting ? (
        <div className="flex-1 flex items-center justify-center text-lg text-gray-600 dark:text-gray-300">
          Connecting to Shruti...
        </div>
      ) : (
        <>
          <ChatWindow
            messages={messages}
            isProcessingAudio={isProcessingAudio}
            isAuroraSpeaking={isAuroraSpeaking}
            playUserAudio={playUserAudio}
            currentlyPlayingUserAudioId={currentlyPlayingUserAudioId}
          />
          {audioPlaybackError && (
            <div className="bg-red-500 text-white text-center p-2 mx-auto max-w-2xl rounded-lg shadow-md mb-2">
              {audioPlaybackError}
            </div>
          )}
          <MessageInput
            input={input}
            onInputChange={setInput}
            onSendText={handleSendText}
            isProcessingAudio={isProcessingAudio}
            isAuroraSpeaking={isAuroraSpeaking}
            isRecording={isRecording}
            onToggleRecording={toggleRecording}
          />
        </>
      )}
    </div>
  );
};


// --- App Mounting Logic ---
const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error("Could not find root element to mount to");
}

const root = ReactDOM.createRoot(rootElement);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
