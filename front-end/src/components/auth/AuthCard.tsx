import React, { useState } from 'react';
import { LoginForm } from './LoginForm';
import { RegistrationForm } from './RegistrationForm';
import { LockKeyhole } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface AuthCardProps {
  initialView?: 'login' | 'register';
}

export const AuthCard: React.FC<AuthCardProps> = ({ initialView  }) => {
  console.log(initialView)
  const [view, setView] = useState<'login' | 'register'>(initialView);
  const navigate = useNavigate();
  return (
    <div className="w-full max-w-md p-8 mx-auto bg-white rounded-xl shadow-lg overflow-hidden transition-all duration-300 ease-in-out">
      <div className="flex justify-center mb-6">
        <div className="w-14 h-14 bg-indigo-600 rounded-full flex items-center justify-center shadow-md">
          <LockKeyhole className="w-7 h-7 text-white" />
        </div>
      </div>
      
      <h2 className="text-2xl font-bold text-center text-gray-800 mb-2">
        {view === 'login' ? 'Attendance-AI' : 'Attendance-AI'}
      </h2>
      <h2 className="text-md font-bold text-center text-gray-800 mb-2">
        {view === 'login' ? 'Teachers Login' : 'Create an account'}
      </h2>
      <p className="text-gray-500 text-center mb-6">
        {view === 'login' 
          ? 'Enter your credentials to access your account' 
          : 'Fill in your details to get started'}
      </p>
      
      <div className="mb-6">
        <div className="flex bg-gray-100 rounded-lg p-1">
          <button
            onClick={() =>{ setView('login'); navigate('/');}}
            className={`flex-1 py-2 text-sm font-medium rounded-md transition-all duration-200 ${
              view === 'login'
                ? 'bg-white text-indigo-600 shadow-sm'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Sign In
          </button>
          <button
            onClick={() =>{ setView('register'); navigate('/signup');} }
            className={`flex-1 py-2 text-sm font-medium rounded-md transition-all duration-200 ${
              view === 'register'
                ? 'bg-white text-indigo-600 shadow-sm'
                : 'text-gray-500 hover:text-gray-700'
            }`}
          >
            Sign Up
          </button>
        </div>
      </div>
      
      <div className="transition-all duration-300 ease-in-out">
        {view === 'login' ? <LoginForm /> : <RegistrationForm onSuccess={() => setView('login')} />}
      </div>
    </div>
  );
};