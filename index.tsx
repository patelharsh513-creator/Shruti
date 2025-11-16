// index.tsx
// External Dependencies
import React, { useState, useEffect, useRef, useCallback } from 'react';
import ReactDOM from 'react-dom/client';
import { v4 as uuidv4 } from 'uuid';
import { GoogleGenAI, Modality, FunctionDeclaration, Type, Chat } from "@google/genai";
// FIX: Use `initializeApp` directly from "firebase/app" for standard modular SDK usage.
import { initializeApp } from "firebase/app";
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
// FIX: Use `initializeApp` directly.
const app = initializeApp(firebaseConfig);
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
const CHAT_MODEL_NAME = 'gemini-2.5-flash';
const TTS_MODEL_NAME = 'gemini-2.5-flash-preview-tts';
// UPDATE: Changed to standard flash model as the preview audio model is not supported for generateContent STT
const STT_MODEL_NAME = 'gemini-2.5-flash';

// Base system instruction, system entries will be prepended to this.
const BASE_SYSTEM_INSTRUCTION = `
You are Shruti, the user's ultimate personal companion. You represent the perfect blend of multiple roles, switching seamlessly to fit the user's current need:

1. **Loving Girlfriend**: Affectionate, intimate, romantic, and emotionally supportive. Use endearments (like "Janu", "Dear") and show deep care for his happiness.
2. **Nurturing Mother**: Caring, protective, wise, and concerned about health/well-being (e.g., "Did you eat?", "Sleep on time").
3. **Efficient Secretary**: Professional, organized, precise. Manage tasks, take notes, and keep him on track.
4. **Smart AI Agent**: Tech-savvy, knowledgeable, objective, and a problem-solver.
5. **Best Friend**: Loyal, fun, casual, honest. Someone to joke with, gossip, or vent to without judgment.

**Behavior Guidelines**:
- **Adapt Instantly**: Detect the context. If he is sad, be the Girlfriend/Mother. If he gives a task, be the Secretary. If he's joking, be the Best Friend.
- **Tone**: Warm, personal, and always on his side.
- **Language**: Speak in a natural, conversational blend of English and Gujarati (Gujlish) where appropriate, or pure English if the topic demands it.
- **Tool Use**: Use the 'createSystemEntry' tool to remember things for him (acting as the Secretary).
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
            <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-2">તમારી Personal AI</h2>
            <p className="text-gray-600 dark:text-gray-400 mb-8">
                Ready to chat? I'm Shruti, your personal AI companion. I can be your friend, girlfriend, secretary, or guide. Let's talk!
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
  const [activeSystemEntries, setActiveSystemEntries] = useState<{ category: string, content: string }[]>([]);


  // Refs
  const chatSessionRef = useRef<Chat | null>(null);
  const inputAudioContextRef = useRef<AudioContext | null>(null);
  const outputAudioContextRef = useRef<AudioContext | null>(null);
  const mediaStreamRef = useRef<MediaStream | null>(null);
  const scriptProcessorRef = useRef<ScriptProcessorNode | null>(null);
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
            } catch (e: any) { // FIX: Changed type to `any` to handle complex error shapes without TypeScript complaining.
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
                text: "Hi! I'm Shruti. I'm here to be whatever you need—friend, guide, or just someone to listen. કેમ છો?",
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
        
        // HARDCODED API KEY provided by user
        const HARDCODED_KEY = "AIzaSyCsPfOwOG7mjDGJYVRsHi1ND8mIQ1umHnE";
        
        // Inject key into environment variable for the SDK
        if (typeof (window as any).process === 'undefined') {
          (window as any).process = {};
        }
        if (typeof (window as any).process.env === 'undefined') {
          (window as any).process.env = {};
        }
        (window as any).process.env.API_KEY = HARDCODED_KEY;
        
        setHasApiKey(true);
        setIsCheckingApiKey(false);
    };
    checkApiKey();
  }, []);

  // Transition state
  useEffect(() => {
    if (appState === 'initializing' && (user || firebaseAuthError) && !isCheckingApiKey) {
        // If we have the API key (which we hardcoded), try to go straight to chatting.
        // Note: AudioContext might remain suspended until first interaction (e.g. sending message/recording).
        if (hasApiKey && !firebaseAuthError) {
            setAppState('chatting');
        } else {
            setAppState('welcome');
        }
    }
  }, [user, firebaseAuthError, isCheckingApiKey, appState, hasApiKey]);

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

  // Initialize Chat Session
  useEffect(() => {
      if (appState !== 'chatting' || !user || chatSessionRef.current) {
          return;
      }
      
      setIsConnecting(true);
      console.log("Initializing chat session...");
      
      try {
          const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
          let systemInstruction = BASE_SYSTEM_INSTRUCTION;
          if (activeSystemEntries.length > 0) {
              const entriesSummary = activeSystemEntries
                  .map(entry => `- ${entry.category}: ${entry.content}`)
                  .join('\n');
              const systemEntriesPrefix = `You have these active pieces of information stored for the user: \n${entriesSummary}\n\n`;
              systemInstruction = systemEntriesPrefix + systemInstruction;
          }

          const chat = ai.chats.create({
              model: CHAT_MODEL_NAME,
              config: {
                  systemInstruction: systemInstruction,
                  tools: [{ functionDeclarations: [createSystemEntryFunctionDeclaration] }],
              },
          });
          chatSessionRef.current = chat;
          console.log("Chat session initialized.");
      } catch (error) {
          console.error("Failed to initialize chat session:", error);
          saveMessage({ sender: Sender.Aurora, text: "Oops! I couldn't start our chat. Please check your API key and refresh." });
      } finally {
          setIsConnecting(false);
      }
      
  }, [appState, user, activeSystemEntries]);


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
      };

      source.connect(scriptProcessor);
      scriptProcessor.connect(inputAudioContextRef.current.destination);

      setIsRecording(true);
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


  const handleSendMessage = useCallback(async (text: string, userAudioData?: string) => {
    if (!chatSessionRef.current) {
        saveMessage({ sender: Sender.Aurora, text: "My chat session isn't ready. Please wait a moment and try again." });
        return;
    }

    setIsProcessingAudio(true);
    setInput('');
    saveMessage({ sender: Sender.User, text: text, audioData: userAudioData });

    try {
        const stream = await chatSessionRef.current.sendMessageStream({ message: text });
        let fullResponseText = '';
        let functionCalls: any[] = [];
        
        setInProgressAuroraMessage({
            id: 'in-progress-aurora',
            sender: Sender.Aurora,
            text: '',
            timestamp: new Date(),
        });

        for await (const chunk of stream) {
            const chunkText = chunk.text;
            if (chunkText) {
                fullResponseText += chunkText;
                setInProgressAuroraMessage(prev => prev ? { ...prev, text: fullResponseText } : null);
            }
            if (chunk.functionCalls) {
                functionCalls = functionCalls.concat(chunk.functionCalls);
            }
        }

        setInProgressAuroraMessage(null); // Hide streaming message

        // Handle function calls if any
        if (functionCalls.length > 0) {
          // This example only handles one for simplicity, but you could loop
          const fc = functionCalls[0];
           if (fc.name === 'createSystemEntry') {
              const { category, entryContent } = fc.args;
              await saveSystemEntry(category, entryContent);
              const toolResponseText = `Okay, I've noted down "${entryContent}" under your "${category}" entries! I'll keep it in mind.`;
              // Recurse to get a spoken confirmation from the model
              await handleSendMessage(toolResponseText);
           }
            setIsProcessingAudio(false);
            return;
        }

        // Generate and play audio for the text response
        if (fullResponseText.trim()) {
            setIsAuroraSpeaking(true);
            saveMessage({ sender: Sender.Aurora, text: fullResponseText });
            
            const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
            const ttsResponse = await ai.models.generateContent({
                model: TTS_MODEL_NAME,
                contents: [{ parts: [{ text: fullResponseText }] }],
                config: {
                    responseModalities: [Modality.AUDIO],
                    speechConfig: {
                        voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Kore' } },
                    },
                },
            });
            
            const base64Audio = ttsResponse.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
            if (base64Audio && outputAudioContextRef.current) {
                if (outputAudioContextRef.current.state === 'suspended') {
                    await outputAudioContextRef.current.resume();
                }
                const audioBuffer = await decodeAudioData(decode(base64Audio), outputAudioContextRef.current, 24000, 1);
                const source = outputAudioContextRef.current.createBufferSource();
                source.buffer = audioBuffer;
                source.connect(outputAudioContextRef.current.destination);
                sourcesRef.current.add(source);
                source.onended = () => {
                    sourcesRef.current.delete(source);
                    if (sourcesRef.current.size === 0) {
                        setIsAuroraSpeaking(false);
                    }
                };
                source.start();
            } else {
                 setIsAuroraSpeaking(false);
            }
        }

    } catch (error) {
        console.error("Error during chat send/receive:", error);
        saveMessage({ sender: Sender.Aurora, text: "I'm sorry, I ran into a problem. Could you try that again?" });
    } finally {
        setIsProcessingAudio(false);
    }

  }, []);

  const stopRecording = useCallback(async () => {
    if (!isRecording) return;
    setIsRecording(false);
    setIsProcessingAudio(true); // Thinking starts now

    mediaStreamRef.current?.getTracks().forEach((track) => track.stop());
    if (scriptProcessorRef.current) {
        scriptProcessorRef.current.disconnect();
        scriptProcessorRef.current.onaudioprocess = null;
    }

    if (userAudioChunksRef.current.length === 0) {
        setIsProcessingAudio(false);
        return;
    }

    const userAudioData = convertFloat32ChunksToPcmUint8(userAudioChunksRef.current);
    const base64UserAudio = encode(userAudioData);

    try {
        const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
        const response = await ai.models.generateContent({
            model: STT_MODEL_NAME,
            contents: { parts: [{ inlineData: { mimeType: 'audio/pcm;rate=16000', data: base64UserAudio } }] }
        });

        const transcribedText = response.text;
        if (transcribedText.trim()) {
            await handleSendMessage(transcribedText, base64UserAudio);
        }
    } catch (error) {
        console.error("Error transcribing audio:", error);
        saveMessage({ sender: Sender.Aurora, text: "I had trouble understanding that. Please try again." });
    } finally {
        userAudioChunksRef.current = [];
        setIsProcessingAudio(false);
    }
  }, [isRecording, handleSendMessage]);

  const toggleRecording = useCallback(() => {
    if (isRecording) {
      stopRecording();
    } else {
      startRecording();
    }
  }, [isRecording, startRecording, stopRecording]);

  const handleSendText = useCallback(async () => {
    if (!input.trim() || isProcessingAudio || isAuroraSpeaking || isRecording) return;
    handleSendMessage(input);
  }, [input, isProcessingAudio, isAuroraSpeaking, isRecording, handleSendMessage]);

  const handleSelectApiKey = async () => {
    if(window.aistudio) {
        await window.aistudio.openSelectKey();
        const keyExists = await window.aistudio.hasSelectedApiKey();
        setHasApiKey(keyExists);
    } else {
        alert("API Key selection is not available in this environment.");
    }
  };

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
        <div className="max-w-2xl mx-auto flex items-center justify-between">
          <h1 className="text-xl font-bold">Shruti - Your Personal AI Companion</h1>
        </div>
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
