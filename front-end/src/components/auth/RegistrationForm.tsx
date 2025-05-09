import React, { useState } from 'react';
import { Input } from '../ui/Input';
import { Button } from '../ui/Button';
import { Eye, EyeOff, User, Mail, KeyRound, CheckCircle2, AlertCircle } from 'lucide-react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';

interface RegistrationFormProps {
  onSuccess?: () => void;
}

export const RegistrationForm: React.FC<RegistrationFormProps> = ({ onSuccess }) => {
  const [isLoading, setIsLoading] = useState(false);
  const [showPassword, setShowPassword] = useState(false);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: '',
    agreeToTerms: false
  });
  const [errors, setErrors] = useState({
    name: '',
    email: '',
    password: '',
    agreeToTerms: ''
  });
  const navigate = useNavigate();
  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : value
    }));
    
    // Clear error when user starts typing
    if (errors[name as keyof typeof errors]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const validatePassword = (password: string) => {
    // At least 8 characters, 1 uppercase, 1 lowercase, 1 number
    return password.length >= 8 && 
           /[A-Z]/.test(password) && 
           /[a-z]/.test(password) && 
           /[0-9]/.test(password);
  };

  const getPasswordStrength = (password: string): { strength: number; text: string; color: string } => {
    if (!password) return { strength: 0, text: 'No password', color: 'bg-gray-200' };
    
    let strength = 0;
    if (password.length >= 8) strength += 1;
    if (/[A-Z]/.test(password)) strength += 1;
    if (/[a-z]/.test(password)) strength += 1;
    if (/[0-9]/.test(password)) strength += 1;
    if (/[^A-Za-z0-9]/.test(password)) strength += 1;
    
    const strengthMap = [
      { text: 'Very weak', color: 'bg-red-500' },
      { text: 'Weak', color: 'bg-orange-500' },
      { text: 'Fair', color: 'bg-yellow-500' },
      { text: 'Good', color: 'bg-blue-500' },
      { text: 'Strong', color: 'bg-green-500' }
    ];
    
    return { 
      strength, 
      ...strengthMap[Math.min(strength, 4)]
    };
  };

  const passwordStrength = getPasswordStrength(formData.password);

  const validate = () => {
    let valid = true;
    const newErrors = { name: '', email: '', password: '', agreeToTerms: '' };
    
    if (!formData.name.trim()) {
      newErrors.name = 'Name is required';
      valid = false;
    }
    
    if (!formData.email) {
      newErrors.email = 'Email is required';
      valid = false;
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Email is invalid';
      valid = false;
    }
    
    if (!formData.password) {
      newErrors.password = 'Password is required';
      valid = false;
    } else if (!validatePassword(formData.password)) {
      newErrors.password = 'Password must be at least 8 characters with uppercase, lowercase, and number';
      valid = false;
    }
    
    if (!formData.agreeToTerms) {
      newErrors.agreeToTerms = 'You must agree to the terms and conditions';
      valid = false;
    }
    
    setErrors(newErrors);
    return valid;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validate()) return;
    
    setIsLoading(true);
    // console.log(formData)
    // signup handle
    try {
      const response = await axios.post(`${import.meta.env.VITE_API}/signup`,formData);
      
      if(response.data.msg=='success')
      {
        alert("account created")
        // navigate(`/`)
        window.location.href = '/';
      }
    } catch (error) {
      console.log(error)
    }finally {
      setIsLoading(false);
    }

    // try {
    //   await new Promise(resolve => setTimeout(resolve, 1500));
    //   console.log('Registration form submitted:', formData);
    //   onSuccess?.();
    //   // Success handling would go here
    // } catch (error) {
    //   console.error('Registration error:', error);
    //   // Error handling would go here
    // } finally {
    //   setIsLoading(false);
    // }
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-5">
      <Input
        label="Full Name"
        name="name"
        type="text"
        icon={<User className="w-5 h-5 text-gray-400" />}
        value={formData.name}
        onChange={handleChange}
        error={errors.name}
        placeholder="John Doe"
        autoComplete="name"
        required
      />
      
      <Input
        label="Email"
        name="email"
        type="email"
        icon={<Mail className="w-5 h-5 text-gray-400" />}
        value={formData.email}
        onChange={handleChange}
        error={errors.email}
        placeholder="your.email@example.com"
        autoComplete="email"
        required
      />
      
      <div className="space-y-2">
        <Input
          label="Password"
          name="password"
          type={showPassword ? 'text' : 'password'}
          icon={<KeyRound className="w-5 h-5 text-gray-400" />}
          value={formData.password}
          onChange={handleChange}
          error={errors.password}
          placeholder="••••••••"
          autoComplete="new-password"
          required
          endIcon={
            <button
              type="button"
              onClick={() => setShowPassword(!showPassword)}
              className="text-gray-400 hover:text-gray-500 focus:outline-none"
            >
              {showPassword ? (
                <EyeOff className="w-5 h-5" />
              ) : (
                <Eye className="w-5 h-5" />
              )}
            </button>
          }
        />
        
        {formData.password && (
          <div className="space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="text-gray-500">Password strength:</span>
              <span className={`font-medium ${
                passwordStrength.strength < 3 ? 'text-red-500' : 
                passwordStrength.strength < 4 ? 'text-yellow-500' : 'text-green-500'
              }`}>
                {passwordStrength.text}
              </span>
            </div>
            
            <div className="h-1.5 w-full bg-gray-200 rounded-full overflow-hidden">
              <div 
                className={`h-full ${passwordStrength.color} transition-all duration-300 ease-in-out`}
                style={{ width: `${(passwordStrength.strength / 5) * 100}%` }}
              ></div>
            </div>
            
            <ul className="space-y-1 mt-2">
              <PasswordRequirement 
                text="At least 8 characters" 
                met={formData.password.length >= 8} 
              />
              <PasswordRequirement 
                text="Contains uppercase letter" 
                met={/[A-Z]/.test(formData.password)} 
              />
              <PasswordRequirement 
                text="Contains lowercase letter" 
                met={/[a-z]/.test(formData.password)} 
              />
              <PasswordRequirement 
                text="Contains number" 
                met={/[0-9]/.test(formData.password)} 
              />
            </ul>
          </div>
        )}
      </div>
      
      <div className="space-y-3">
        <div className="flex items-start">
          <div className="flex items-center h-5">
            <input
              id="agreeToTerms"
              name="agreeToTerms"
              type="checkbox"
              checked={formData.agreeToTerms}
              onChange={handleChange}
              className="w-4 h-4 text-indigo-600 border-gray-300 rounded focus:ring-indigo-500"
            />
          </div>
          <div className="ml-3 text-sm">
            <label htmlFor="agreeToTerms" className="font-medium text-gray-700">
              I agree to the <a href="#" className="text-indigo-600 hover:text-indigo-500">Terms of Service</a> and <a href="#" className="text-indigo-600 hover:text-indigo-500">Privacy Policy</a>
            </label>
            {errors.agreeToTerms && (
              <p className="text-red-500 text-xs mt-1">{errors.agreeToTerms}</p>
            )}
          </div>
        </div>
      </div>
      
      <Button type="submit" isLoading={isLoading} fullWidth>
        Create account
      </Button>
    </form>
  );
};

interface PasswordRequirementProps {
  text: string;
  met: boolean;
}

const PasswordRequirement: React.FC<PasswordRequirementProps> = ({ text, met }) => (
  <li className={`text-xs flex items-center ${met ? 'text-green-600' : 'text-gray-500'}`}>
    {met ? (
      <CheckCircle2 className="w-3.5 h-3.5 mr-1.5 flex-shrink-0" />
    ) : (
      <AlertCircle className="w-3.5 h-3.5 mr-1.5 flex-shrink-0" />
    )}
    {text}
  </li>
);