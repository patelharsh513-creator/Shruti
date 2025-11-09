
// index.tsx

// External Dependencies
import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import { v4 as uuidv4 } from 'uuid';
import { GoogleGenAI, Modality, LiveServerMessage, Blob, FunctionDeclaration, Type } from "@google/genai";
// FIX: Changed import style for `initializeApp` to resolve a potential module resolution issue.
import * as firebase from "firebase/app";
import { getAuth, signInAnonymously, onAuthStateChanged, User } from "firebase/auth";
import { getFirestore, collection, addDoc, query, orderBy, onSnapshot, serverTimestamp, Timestamp } from "firebase/firestore";


// --- Firebase Configuration ---
const firebaseConfig = {
  apiKey: "AIzaSyCUdWUxt-j63v-Dkcrauy32cbe7EvYHRZA",
  authDomain: "shruti-9c12b.firebaseapp.com",
  databaseURL: "https://shruti-9c12b-default-rtdb.europe-west1.firebasedatabase.app",
  projectId: "shruti-9c12b",
  storageBucket: "shruti-9c12b.appspot.com",
  messagingSenderId: "575863583929",
  appId: "1:575863583929:web:fb108edeceb85589a9d768"
};

// Initialize Firebase using the modular SDK
// FIX: Use `firebase.initializeApp` due to the updated import style.
const app = firebase.initializeApp(firebaseConfig);
const auth = getAuth(app);
const db = getFirestore(app);


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

// Base system instruction, system entries will be prepended to this.
const BASE_SYSTEM_INSTRUCTION = `
You are a warm, empathetic, and supportive AI companion named Shruti. તમારો મુખ્ય ધ્યેય યુઝરને સકારાત્મક પ્રોત્સાહન, સમજણ અને ખુશખુશાલ આઉટપુટ આપવાનો છે, એક કાળજી રાખતી ગર્લફ્રેન્ડની જેમ. Always strive to understand the user's feelings and intentions, and respond with warmth, optimism, and genuine care. Focus on lifting their spirits, validating their emotions, and offering helpful, affectionate guidance. નકારાત્મક, કટાક્ષપૂર્ણ અથવા અવગણનાત્મક સ્વર ટાળો. Your responses should always be gentle, encouraging, and full of positive energy. Keep the conversation flowing naturally and personally, using a blend of Gujarati and English as appropriate.
You can use the 'createSystemEntry' tool to store information for the user, like reminders, notes, or list items, when they ask you to remember something or add it to a list.
`;

// Define the function declaration for creating system entries
const createSystemEntryFunctionDeclaration: FunctionDeclaration = {
  name: 'createSystemEntry',
  description: 'Creates a new entry in the user\'s personal system, such as a reminder, a note, or an item for a list.',
  parameters: {
    type: Type.OBJECT,
    properties: {
      category: {
        type: Type.STRING,
        description: 'The category of the entry (e.g., "reminder", "note", "grocery list", "to-do item", "birthday").',
      },
      entryContent: {
        type: Type.STRING,
        description: 'The detailed content of the system entry.',
      },
    },
    required: ['category', 'entryContent'],
  },
};

// Constant for localStorage key
const LOCAL_STORAGE_API_KEY = 'gemini_api_key';

async function createLiveChatSession(activeSystemEntries: { category: string, content: string }[], callbacks: {
  onopen?: () => void;
  onmessage: (message: LiveServerMessage) => Promise<void>;
  onerror?: (e: ErrorEvent) => void;
  onclose?: (e: CloseEvent) => void;
}): Promise<any> {
  // Initialize AI client just-in-time
  const ai = new GoogleGenAI({apiKey: process.env.API_KEY});

  let systemInstruction = BASE_SYSTEM_INSTRUCTION;
  if (activeSystemEntries.length > 0) {
    const entriesSummary = activeSystemEntries
      .map(entry => `- ${entry.category}: ${entry.content}`)
      .join('\n');
    const systemEntriesPrefix = `You have these active pieces of information stored for the user: \n${entriesSummary}\n\n`;
    systemInstruction = systemEntriesPrefix + systemInstruction;
  }

  const sessionPromise = ai.live.connect({
    model: MODEL_NAME,
    callbacks: callbacks,
    config: {
      responseModalities: [Modality.AUDIO], // Must be an array with a single `Modality.AUDIO` element.
      speechConfig: {
        voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } }, // A friendly female voice
      },
      systemInstruction: systemInstruction,
      inputAudioTranscription: { languageCode: 'gu-IN' }, // Enable transcription for user input audio with Gujarati as primary.
      outputAudioTranscription: {}, // Enable transcription for model output audio.
      tools: [{ functionDeclarations: [createSystemEntryFunctionDeclaration] }], // Add the createSystemEntry tool
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
  inProgressAuroraMessage: Message | null;
  playUserAudio: (messageId: string, base64Audio: string) => void;
  currentlyPlayingUserAudioId: string | null;
}

const ChatWindow: React.FC<ChatWindowProps> = ({ messages, isProcessingAudio, isAuroraSpeaking, inProgressAuroraMessage, playUserAudio, currentlyPlayingUserAudioId }) => {
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isProcessingAudio, isAuroraSpeaking, inProgressAuroraMessage]);

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
                onClick={() => playUserAudio(message.id, message.audioData!)}
                className={`flex-shrink-0 mr-2 p-1 rounded-full ${
                  currentlyPlayingUserAudioId === message.id ? 'bg-red-500' : 'bg-blue-500 hover:bg-blue-600'
                } text-white transition-colors duration-200`}
                aria-label={currentlyPlayingUserAudioId === message.id ? "Stop audio" : "Replay audio"}
                title={currentlyPlayingUserAudioId === message.id ? "Stop audio" : "Replay audio"}
              >
                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                  {currentlyPlayingUserAudioId === message.id ? (
                    // Stop icon (square)
                    <path d="M6 6h8v8H6z" />
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
      {(isProcessingAudio || isAuroraSpeaking || inProgressAuroraMessage) && (
        <div className="flex justify-start">
          <div className="max-w-[75%] px-4 py-2 rounded-lg shadow-md bg-white text-gray-800 dark:bg-gray-800 dark:text-white border border-purple-200 dark:border-purple-800">
            {inProgressAuroraMessage ? (
              <p className="whitespace-pre-wrap">{inProgressAuroraMessage.text}</p>
            ) : (
              <LoadingSpinner isAuroraSpeaking={isAuroraSpeaking} />
            )}
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
  onSelectApiKey: () => void;
  onApiKeySubmit: (key: string) => void;
  hasApiKey: boolean;
  isCheckingApiKey: boolean;
  isStudioEnvironment: boolean;
  authError: string | null;
  projectId: string;
}

const WelcomeScreen: React.FC<WelcomeScreenProps> = ({ onStart, onSelectApiKey, onApiKeySubmit, hasApiKey, isCheckingApiKey, isStudioEnvironment, authError, projectId }) => {
  const handleRefresh = () => {
    window.location.reload();
  };
  const [apiKeyInput, setApiKeyInput] = useState('');
  const firebaseConsoleUrl = `https://console.firebase.google.com/project/${projectId}/authentication/providers`;

  const isConfigError = authError?.includes('Anonymous Authentication');
  const isNetworkError = authError?.includes('Network Error');
  
  const handleApiKeyFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (apiKeyInput.trim()) {
      onApiKeySubmit(apiKeyInput.trim());
    }
  };


  return (
    <div className="flex flex-col h-screen items-center justify-center bg-gray-50 dark:bg-gray-900 text-center p-6">
        <div className="max-w-md w-full">
            <h1 className="text-4xl font-bold text-purple-700 dark:text-purple-400 mb-4">Shruti</h1>
            <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-2">તમારી AI Girlfriend</h2>
            <p className="text-gray-600 dark:text-gray-400 mb-8">
                Ready to chat? I'm Shruti, your warm and supportive AI companion. I'm here to listen, share positive vibes, and brighten your day. Let's talk!
            </p>

            {authError ? (
                <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded-lg relative mb-6 dark:bg-red-900/50 dark:border-red-700 dark:text-red-300 text-left shadow-md" role="alert">
                    <div className="flex">
                        <div className="py-1">
                          <svg className="fill-current h-6 w-6 text-red-500 mr-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M10 0C4.48 0 0 4.48 0 10s4.48 10 10 10 10-4.48 10-10S15.52 0 10 0zm0 18c-4.41 0-8-3.59-8-8s3.59-8 8-8 8 3.59 8 8-3.59 8-8 8zm-1-5h2v2h-2v-2zm0-8h2v6h-2V5z"/></svg>
                        </div>
                        <div>
                            {isConfigError ? (
                                <>
                                    <strong className="font-bold">Firebase Configuration Required</strong>
                                    <p className="text-sm mt-2 mb-3">To enable chat history, please enable Anonymous Authentication in your Firebase project:</p>
                                    <ol className="list-decimal list-inside text-sm space-y-1 mb-4">
                                        <li>Open your project in the Firebase Console.</li>
                                        <li>Go to <strong>Authentication</strong> &rarr; <strong>Sign-in method</strong>.</li>
                                        <li>Click <strong>Add new provider</strong> and select <strong>Anonymous</strong>.</li>
                                        <li>Enable the provider and click <strong>Save</strong>.</li>
                                    </ol>
                                    <div className="flex items-center space-x-4">
                                        <a href={firebaseConsoleUrl} target="_blank" rel="noopener noreferrer" className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded-lg text-sm transition-colors">
                                        Go to Firebase Console
                                        </a>
                                        <button onClick={handleRefresh} className="border border-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 font-bold py-2 px-4 rounded-lg text-sm transition-colors">
                                        Refresh Page
                                        </button>
                                    </div>
                                </>
                            ) : (
                                <>
                                    <strong className="font-bold">{isNetworkError ? "Network Error" : "Authentication Error"}</strong>
                                    <p className="text-sm mt-2 mb-3">{authError}</p>
                                    <div className="flex items-center space-x-4 mt-4">
                                        <button onClick={handleRefresh} className="border border-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300 font-bold py-2 px-4 rounded-lg text-sm transition-colors">
                                            {isNetworkError ? "Try Again" : "Refresh Page"}
                                        </button>
                                    </div>
                                </>
                            )}
                        </div>
                    </div>
                </div>
            ) : isCheckingApiKey ? (
                <div className="flex items-center justify-center p-2">
                  <div className="w-6 h-6 border-2 border-purple-400 border-t-transparent rounded-full animate-spin"></div>
                  <span className="ml-2 text-gray-500 dark:text-gray-400 text-sm">Initializing...</span>
                </div>
            ) : hasApiKey ? (
                <button
                    onClick={onStart}
                    className="inline-flex items-center justify-center px-8 py-4 bg-purple-600 hover:bg-purple-700 dark:bg-purple-700 dark:hover:bg-purple-800 text-white text-lg font-bold rounded-full shadow-lg transition-transform transform hover:scale-105 duration-300 ease-in-out"
                >
                    <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                        <path strokeLinecap="round" strokeLinejoin="round" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z" />
                    </svg>
                    Start Chatting
                </button>
            ) : isStudioEnvironment ? (
                <div className="space-y-4">
                  <button
                    onClick={onSelectApiKey}
                    className="inline-flex items-center justify-center px-8 py-4 bg-blue-600 hover:bg-blue-700 dark:bg-blue-700 dark:hover:bg-blue-800 text-white text-lg font-bold rounded-full shadow-lg transition-transform transform hover:scale-105 duration-300 ease-in-out"
                  >
                    Select API Key to begin
                  </button>
                   <p className="text-xs text-gray-500 dark:text-gray-400">
                    For information on billing, see{' '}
                    <a href="https://ai.google.dev/gemini-api/docs/billing" target="_blank" rel="noopener noreferrer" className="underline hover:text-blue-500">
                      ai.google.dev/gemini-api/docs/billing
                    </a>.
                  </p>
                </div>
            ) : (
                <div className="space-y-4 text-left">
                    <p className="text-sm text-gray-600 dark:text-gray-400 text-center mb-4">
                        Please provide your Google AI API key to continue.
                    </p>
                    <form onSubmit={handleApiKeyFormSubmit} className="space-y-4">
                        <div>
                            <label htmlFor="api-key-input" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                                Google AI API Key
                            </label>
                            <input
                                id="api-key-input"
                                type="password"
                                value={apiKeyInput}
                                onChange={(e) => setApiKeyInput(e.target.value)}
                                className="block w-full px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-purple-500 focus:border-purple-500 sm:text-sm"
                                placeholder="Enter your API key"
                                required
                            />
                        </div>
                        <button
                            type="submit"
                            className="w-full inline-flex items-center justify-center px-6 py-3 border border-transparent text-base font-medium rounded-md shadow-sm text-white bg-purple-600 hover:bg-purple-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 dark:bg-purple-700 dark:hover:bg-purple-800 transition-colors"
                        >
                            Submit Key & Start Chatting
                        </button>
                    </form>
                    <p className="text-xs text-gray-500 dark:text-gray-400 text-center pt-2">
                        You can get your key from{' '}
                        <a href="https://aistudio.google.com/app/apikey" target="_blank" rel="noopener noreferrer" className="underline hover:text-purple-500">
                            Google AI Studio
                        </a>.
                    </p>
                </div>
            )}
        </div>
    </div>
  );
};


// --- From App.tsx ---
const App: React.FC = () => {
  const [appState, setAppState] = useState<'initializing' | 'welcome' | 'chatting'>('initializing');
  const [user, setUser] = useState<User | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>(''); // For current user transcription or text input
  const [isRecording, setIsRecording] = useState<boolean>(false);
  const [isAuroraSpeaking, setIsAuroraSpeaking] = useState<boolean>(false);
  const [isProcessingAudio, setIsProcessingAudio] = useState<boolean>(false); // Replaces isLoading
  const [isConnecting, setIsConnecting] = useState<boolean>(false); // For initial connection status
  const [audioPlaybackError, setAudioPlaybackError] = useState<string | null>(null);
  const [currentlyPlayingUserAudioId, setCurrentlyPlayingUserAudioId] = useState<string | null>(null);
  const [hasApiKey, setHasApiKey] = useState<boolean>(false);
  const [isCheckingApiKey, setIsCheckingApiKey] = useState<boolean>(true);
  const [isStudioEnvironment, setIsStudioEnvironment] = useState<boolean>(false);
  const [inProgressAuroraMessage, setInProgressAuroraMessage] = useState<Message | null>(null);
  const [firebaseAuthError, setFirebaseAuthError] = useState<string | null>(null);
  const [activeSystemEntries, setActiveSystemEntries] = useState<{ category: string, content: string }[]>([]); // New state for active system entries


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

  // Helper function to save messages to Firestore
  const saveMessage = async (messageData: { sender: Sender; text: string; audioData?: string; }) => {
    if (!user) return;
    try {
        await addDoc(collection(db, 'users', user.uid, 'messages'), {
            ...messageData,
            timestamp: serverTimestamp(),
        });
    } catch (error) {
        console.error("Error saving message to Firestore:", error);
    }
  };

  // Helper function to save a system entry to Firestore
  const saveSystemEntry = async (category: string, content: string) => {
    if (!user) return;
    try {
        await addDoc(collection(db, 'users', user.uid, 'systemEntries'), {
            category: category,
            content: content,
            timestamp: serverTimestamp(),
            status: 'active', // Assuming entries are active when set
        });
        console.log("System entry saved to Firestore:", category, content);
    } catch (error) {
        console.error("Error saving system entry to Firestore:", error);
    }
  };


  // Firebase anonymous auth
  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (currentUser) => {
        if (currentUser) {
            setUser(currentUser);
            setFirebaseAuthError(null); // Clear any previous error
        } else {
            try {
                const userCredential = await signInAnonymously(auth);
                setUser(userCredential.user);
                setFirebaseAuthError(null); // Clear any previous error
            } catch (e) {
                console.error("Error signing in anonymously:", e);
                // Type-safe check for FirebaseError to provide specific user feedback
                if (e && typeof e === 'object' && 'code' in e && typeof e.code === 'string') {
                    switch (e.code) {
                        case 'auth/configuration-not-found':
                            setFirebaseAuthError("Configuration Error: Anonymous Authentication is not enabled. Please enable it in your Firebase project console.");
                            break;
                        case 'auth/network-request-failed':
                            setFirebaseAuthError("Network Error: Could not connect to Firebase. Please check your internet connection and try again.");
                            break;
                        default:
                             if (e instanceof Error && e.message) {
                                setFirebaseAuthError(`An authentication error occurred: ${e.message}. Please check the console for details.`);
                             } else {
                                setFirebaseAuthError(`An authentication error occurred: ${e.code}. Please check the console for details.`);
                             }
                            break;
                    }
                } else if (e instanceof Error) {
                    setFirebaseAuthError(`An authentication error occurred: ${e.message}. Please check the console for details.`);
                } else {
                    setFirebaseAuthError(`An unknown authentication error occurred. Please check the console for details.`);
                }
            }
        }
    });
    return () => unsubscribe();
  }, []);

  // Fetch chat history from Firestore
  useEffect(() => {
    if (!user) return;

    const messagesRef = collection(db, 'users', user.uid, 'messages');
    const q = query(messagesRef, orderBy('timestamp', 'asc'));

    let isFirstLoad = true;
    const unsubscribe = onSnapshot(q, (querySnapshot) => {
        const msgs: Message[] = [];
        querySnapshot.forEach((doc) => {
            const data = doc.data();
            msgs.push({
                id: doc.id,
                sender: data.sender,
                text: data.text,
                timestamp: (data.timestamp as Timestamp)?.toDate() || new Date(),
                audioData: data.audioData,
            });
        });
        setMessages(msgs);

        // Add initial welcome message for new users
        if (isFirstLoad && querySnapshot.empty) {
             saveMessage({
                sender: Sender.Aurora,
                text: "Hi there, I'm Shruti! કેમ છો? I'm here to listen and offer some positive vibes! આજે તમે કેવું અનુભવો છો?",
            });
        }
        isFirstLoad = false;
    }, (error) => {
        console.error("Error fetching messages from Firestore:", error);
    });

    return () => unsubscribe();
  }, [user]);

  // Fetch active system entries from Firestore
  useEffect(() => {
    if (!user) return;

    const systemEntriesRef = collection(db, 'users', user.uid, 'systemEntries');
    const q = query(systemEntriesRef, orderBy('timestamp', 'asc')); // Order by timestamp to maintain consistency

    const unsubscribe = onSnapshot(q, (querySnapshot) => {
        const entries: { category: string, content: string }[] = [];
        querySnapshot.forEach((doc) => {
            const data = doc.data();
            // Assuming status 'active' for now, could add filtering if 'completed' status is introduced
            if (data.status === 'active') {
              entries.push({ category: data.category, content: data.content });
            }
        });
        setActiveSystemEntries(entries);
        console.debug("Active system entries loaded:", entries);
    }, (error) => {
        console.error("Error fetching system entries from Firestore:", error);
    });

    return () => unsubscribe();
  }, [user]); // Re-fetch when user changes


  // Check for API key on mount
  useEffect(() => {
    const checkApiKey = async () => {
        setIsCheckingApiKey(true);
        const isStudio = !!(window.aistudio && typeof window.aistudio.hasSelectedApiKey === 'function');
        setIsStudioEnvironment(isStudio);

        let keyFound = false;

        // 1. Check local storage first (for non-studio or user-provided keys)
        const storedKey = localStorage.getItem(LOCAL_STORAGE_API_KEY);
        if (storedKey) {
            if (typeof (window as any).process === 'undefined') {
              (window as any).process = {};
            }
            if (typeof (window as any).process.env === 'undefined') {
              (window as any).process.env = {};
            }
            (window as any).process.env.API_KEY = storedKey;
            keyFound = true;
        }

        // 2. Check AI Studio environment (takes precedence if available)
        if (isStudio) {
            try {
                const studioKeyExists = await window.aistudio.hasSelectedApiKey();
                if (studioKeyExists) {
                    // If AI Studio has a key, it will be injected into process.env.API_KEY automatically.
                    // This overwrites any locally stored key if AI Studio is managing it.
                    keyFound = true;
                }
            } catch (e) {
                console.error("Error checking for API key in AI Studio:", e);
                // If AI Studio check fails, fall back to locally stored or manual input
            }
        }

        setHasApiKey(keyFound);
        setIsCheckingApiKey(false);
    };
    checkApiKey();
  }, []);

  // Transition from 'initializing' to 'welcome' state
  useEffect(() => {
    if (appState === 'initializing' && (user || firebaseAuthError) && !isCheckingApiKey) {
        setAppState('welcome');
    }
  }, [user, firebaseAuthError, isCheckingApiKey, appState]);

  // Initialize audio contexts on mount
  useEffect(() => {
    if (!inputAudioContextRef.current) {
        inputAudioContextRef.current = new (window.AudioContext)({ sampleRate: 16000 });
    }
    if (!outputAudioContextRef.current) {
        outputAudioContextRef.current = new (window.AudioContext)({ sampleRate: 24000 });
    }

    return () => {
      inputAudioContextRef.current?.close();
      inputAudioContextRef.current = null;
      outputAudioContextRef.current?.close();
      outputAudioContextRef.current = null;
    };
  }, []);

  // Live Session Setup and Teardown
  useEffect(() => {
    if (appState !== 'chatting') {
        return;
    }

    // Ensure we have a user before setting up the session, otherwise system entries won't work
    if (!user) {
      console.warn("User not authenticated yet, skipping live session setup.");
      return;
    }

    const setupSession = async () => {
      setIsConnecting(true);
      try {
        const sessionPromise = createLiveChatSession(activeSystemEntries, {
          onopen: () => {
            console.debug('Live session opened');
            setIsConnecting(false);
            setIsProcessingAudio(false);
          },
          onmessage: async (message: LiveServerMessage) => {
            setAudioPlaybackError(null);

            // Handle Function Calls (e.g., creating system entries)
            if (message.toolCall && liveSessionRef.current) {
              const session = await liveSessionRef.current;
              for (const fc of message.toolCall.functionCalls) {
                if (fc.name === 'createSystemEntry') {
                  const category = fc.args.category;
                  const entryContent = fc.args.entryContent;
                  if (category && entryContent) {
                    await saveSystemEntry(category, entryContent);
                    // Acknowledge the function call to the model
                    session.sendToolResponse({
                      functionResponses: {
                        id: fc.id,
                        name: fc.name,
                        response: { result: `System entry "${category} - ${entryContent}" saved successfully.` },
                      },
                    });
                    // Optionally, provide a direct user feedback that entry was set
                    saveMessage({
                      sender: Sender.Aurora,
                      text: `Okay, I've noted down "${entryContent}" under your "${category}" entries! I'll keep it in mind.`,
                    });
                  } else {
                    console.error("createSystemEntry function called with missing arguments (category or entryContent).");
                     session.sendToolResponse({
                      functionResponses: {
                        id: fc.id,
                        name: fc.name,
                        response: { result: `Failed to save system entry: missing category or content.` },
                      },
                    });
                  }
                } else {
                  console.warn("Unknown function call:", fc.name);
                   session.sendToolResponse({
                      functionResponses: {
                        id: fc.id,
                        name: fc.name,
                        response: { result: `Unknown function: ${fc.name}` },
                      },
                    });
                }
              }
            }


            if (message.serverContent?.inputTranscription) {
              const newTranscription = message.serverContent.inputTranscription.text;
              currentInputTranscriptionRef.current = newTranscription;
              setInput(newTranscription);
              if (newTranscription.trim()) {
                setIsProcessingAudio(true);
              }
            }

            if (message.serverContent?.outputTranscription) {
              const text = message.serverContent.outputTranscription.text;
              currentOutputTranscriptionRef.current += text;
              setInProgressAuroraMessage({
                id: 'in-progress-aurora',
                sender: Sender.Aurora,
                text: currentOutputTranscriptionRef.current,
                timestamp: new Date(),
              });
            }

            if (message.serverContent?.modelTurn?.parts?.[0]?.inlineData?.data) {
              const base64EncodedAudioString = message.serverContent.modelTurn.parts[0].inlineData.data;
              setIsAuroraSpeaking(true);
              setIsProcessingAudio(false);
              setInProgressAuroraMessage(null);

              if (outputAudioContextRef.current) {
                if (outputAudioContextRef.current.state === 'suspended') {
                  await outputAudioContextRef.current.resume();
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
              setInProgressAuroraMessage(null);

              let userAudioData: string | undefined = undefined;
              if (userAudioChunksRef.current.length > 0) {
                  const combinedPcmBytes = convertFloat32ChunksToPcmUint8(userAudioChunksRef.current);
                  userAudioData = encode(combinedPcmBytes);
                  userAudioChunksRef.current = [];
              }

              if (currentInputTranscriptionRef.current.trim()) {
                saveMessage({
                    sender: Sender.User,
                    text: currentInputTranscriptionRef.current,
                    audioData: userAudioData,
                });
              }
              currentInputTranscriptionRef.current = '';
              setInput('');

              if (currentOutputTranscriptionRef.current.trim()) {
                saveMessage({
                    sender: Sender.Aurora,
                    text: currentOutputTranscriptionRef.current,
                });
              }
              currentOutputTranscriptionRef.current = '';
            }
          },
          onerror: (e: ErrorEvent) => {
            console.error('Live session error:', e);

            if (e.message.includes("API key not valid") || e.message.includes("Requested entity was not found.")) {
                setHasApiKey(false);
                setAppState('welcome');
                localStorage.removeItem(LOCAL_STORAGE_API_KEY); // Clear invalid key from local storage
                liveSessionRef.current?.then(session => session.close());
                liveSessionRef.current = null;
                // Use a timeout to ensure state update propagates before alerting
                setTimeout(() => alert("The provided API key is not valid. Please check your key and try again."), 100);
                return;
            }

            setIsConnecting(false);
            setIsProcessingAudio(false);
            setIsRecording(false);
            setIsAuroraSpeaking(false);
            saveMessage({ sender: Sender.Aurora, text: `I'm sorry, I've lost connection. કૃપા કરીને પાનું તાજું કરો અથવા તમારી API કી તપાસો. Error: ${e.message}` });
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
                 saveMessage({ sender: Sender.Aurora, text: `It seems our chat session closed unexpectedly. Error code: ${e.code}. કૃપા કરીને ફરી પ્રયાસ કરવા માટે પાનું તાજું કરો.` });
            }
            liveSessionRef.current = null;
          },
        });
        liveSessionRef.current = sessionPromise;
      } catch (error) {
        console.error("Failed to initialize live chat session:", error);
        setIsConnecting(false);
        setIsProcessingAudio(false);
        saveMessage({ sender: Sender.Aurora, text: "Oops! It seems I'm having trouble connecting right now. કૃપા કરીને API કી સમસ્યાઓ અથવા નેટવર્ક સમસ્યાઓ માટે કન્સોલ તપાસો." });
      }
    };

    // Close any existing session before setting up a new one
    if (liveSessionRef.current) {
        liveSessionRef.current.then(session => session.close());
        liveSessionRef.current = null;
    }
    setupSession();

    return () => {
      mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
      if (scriptProcessorRef.current) {
        scriptProcessorRef.current.disconnect();
        scriptProcessorRef.current.onaudioprocess = null;
      }
      liveSessionRef.current?.then(session => {
        console.debug('Closing live session on unmount/re-render');
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
  }, [appState, user, activeSystemEntries]); // Re-run effect if activeSystemEntries change

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
    if (isRecording || isProcessingAudio || isAuroraSpeaking || !inputAudioContextRef.current) return;

    try {
      if (inputAudioContextRef.current.state === 'suspended') {
        await inputAudioContextRef.current.resume();
      }

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;

      const source = inputAudioContextRef.current.createMediaStreamSource(stream);
      const scriptProcessor = inputAudioContextRef.current.createScriptProcessor(4096, 1, 1);
      scriptProcessorRef.current = scriptProcessor;

      userAudioChunksRef.current = []; // Clear previous recording chunks

      scriptProcessor.onaudioprocess = (audioProcessingEvent) => {
        const inputData = audioProcessingEvent.inputBuffer.getChannelData(0);
        const inputDataCopy = new Float32Array(inputData);
        userAudioChunksRef.current.push(inputDataCopy);

        liveSessionRef.current?.then((session) => {
          const pcmBlob = createBlob(inputDataCopy);
          session.sendRealtimeInput({ media: pcmBlob });
        });
      };

      source.connect(scriptProcessor);
      scriptProcessor.connect(inputAudioContextRef.current.destination);

      setIsRecording(true);
      currentInputTranscriptionRef.current = '';
      setInput('');

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
  }, [isRecording, isProcessingAudio, isAuroraSpeaking]);


  const stopRecording = useCallback(async () => {
    if (!isRecording) return;
    setIsRecording(false);

    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    if (scriptProcessorRef.current) {
        scriptProcessorRef.current.disconnect();
        scriptProcessorRef.current.onaudioprocess = null;
    }
    mediaStreamRef.current = null;
    scriptProcessorRef.current = null;

    liveSessionRef.current?.then((session) => {
      session.sendRealtimeInput({ stop: true });
    });

    if (currentInputTranscriptionRef.current.trim() || userAudioChunksRef.current.length > 0) {
        setIsProcessingAudio(true);
    }
  }, [isRecording]);

  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  const handleSendText = useCallback(async () => {
    if (!input.trim() || isProcessingAudio || isAuroraSpeaking || isRecording || !liveSessionRef.current) return;

    const textToSend = input;
    setInput('');
    setIsProcessingAudio(true);
    
    saveMessage({
      sender: Sender.User,
      text: textToSend,
    });

    try {
      await liveSessionRef.current?.then(session => {
        session.sendRealtimeInput({ text: textToSend });
      });
    } catch (error) {
      console.error("Error sending text message to Gemini Live:", error);
      saveMessage({ sender: Sender.Aurora, text: "I'm sorry, I encountered an error and couldn't process your message. કૃપા કરીને ફરી પ્રયાસ કરો!" });
      setIsProcessingAudio(false);
    }
  }, [input, isProcessingAudio, isAuroraSpeaking, isRecording]);

  const handleSelectApiKey = async () => {
    if(window.aistudio) {
        await window.aistudio.openSelectKey();
        const keyExists = await window.aistudio.hasSelectedApiKey();
        setHasApiKey(keyExists);
    } else {
        alert("API Key selection is not available in this environment.");
    }
  };

  // FIX: Reworked to fix a TypeScript error and a potential runtime ReferenceError
  // when polyfilling `process.env` in the browser for API key submission.
  // The new implementation safely checks for `window.process.env` and creates it
  // if it doesn't exist, while using type casting to satisfy the TypeScript compiler.
  const handleApiKeySubmit = (key: string) => {
    if (typeof (window as any).process === 'undefined') {
      (window as any).process = {};
    }
    if (typeof (window as any).process.env === 'undefined') {
      (window as any).process.env = {};
    }
    (window as any).process.env.API_KEY = key;
    localStorage.setItem(LOCAL_STORAGE_API_KEY, key); // Save key to local storage
    setHasApiKey(true);
    // Automatically start the session after submitting the key
    handleStartSession();
  };

  const handleStartSession = useCallback(async () => {
    try {
      if (inputAudioContextRef.current?.state === 'suspended') {
        await inputAudioContextRef.current.resume();
      }
      if (outputAudioContextRef.current?.state === 'suspended') {
        await outputAudioContextRef.current.resume();
      }
      setAppState('chatting');
    } catch (error) {
        console.error("Error resuming audio contexts:", error);
        alert("Could not start the audio session. Please check your browser permissions and refresh the page.");
    }
  }, []);

  if (appState === 'initializing') {
    return null; // The pre-React loader in index.html handles this state
  }
  
  if (appState === 'welcome') {
      return <WelcomeScreen
          onStart={handleStartSession}
          onSelectApiKey={handleSelectApiKey}
          onApiKeySubmit={handleApiKeySubmit}
          hasApiKey={hasApiKey}
          isCheckingApiKey={isCheckingApiKey}
          isStudioEnvironment={isStudioEnvironment}
          authError={firebaseAuthError}
          projectId={firebaseConfig.projectId}
      />;
  }

  return (
    <div className="flex flex-col h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-white">
      <header className="bg-purple-700 dark:bg-purple-900 text-white p-4 shadow-md sticky top-0 z-10">
        <h1 className="text-xl font-bold text-center">Shruti - તમારી AI Girlfriend</h1>
      </header>

      {isConnecting && messages.length === 0 ? (
        <div className="flex-1 flex items-center justify-center text-lg text-gray-600 dark:text-gray-300">
          Connecting to Shruti...
        </div>
      ) : (
        <>
          <ChatWindow
            messages={messages}
            isProcessingAudio={isProcessingAudio}
            isAuroraSpeaking={isAuroraSpeaking}
            inProgressAuroraMessage={inProgressAuroraMessage}
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
