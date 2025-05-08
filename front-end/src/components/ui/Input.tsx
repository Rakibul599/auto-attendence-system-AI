import React, { useState } from 'react';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label: string;
  error?: string;
  icon?: React.ReactNode;
  endIcon?: React.ReactNode;
}

export const Input: React.FC<InputProps> = ({
  label,
  error,
  icon,
  endIcon,
  className = '',
  ...props
}) => {
  const [isFocused, setIsFocused] = useState(false);
  const hasValue = Boolean(props.value);

  return (
    <div className="relative">
      <div className="relative">
        {icon && (
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            {icon}
          </div>
        )}
        
        <input
          {...props}
          className={`block w-full ${
            icon ? 'pl-10' : 'pl-4'
          } ${
            endIcon ? 'pr-10' : 'pr-4'
          } py-2.5 border ${
            error
              ? 'border-red-500 focus:ring-red-500 focus:border-red-500'
              : 'border-gray-300 focus:ring-indigo-500 focus:border-indigo-500'
          } rounded-md shadow-sm text-gray-900 placeholder-transparent ${className}`}
          placeholder={props.placeholder || ' '}
          onFocus={(e) => {
            setIsFocused(true);
            props.onFocus?.(e);
          }}
          onBlur={(e) => {
            setIsFocused(false);
            props.onBlur?.(e);
          }}
          id={props.name}
        />
        
        {endIcon && (
          <div className="absolute inset-y-0 right-0 pr-3 flex items-center">
            {endIcon}
          </div>
        )}
        
        <label
          htmlFor={props.name}
          className={`absolute text-sm duration-200 transform ${
            (isFocused || hasValue) 
              ? '-translate-y-2 scale-90 top-0 z-10 ml-3 bg-white px-1 py-0'
              : 'top-2.5'
          } ${
            icon ? 'left-9' : 'left-3'
          } ${
            isFocused
              ? 'text-indigo-600'
              : error
              ? 'text-red-500'
              : 'text-gray-500'
          } origin-[0] pointer-events-none`}
        >
          {label}
        </label>
      </div>
      
      {error && (
        <p className="mt-1 text-sm text-red-600">{error}</p>
      )}
    </div>
  );
};