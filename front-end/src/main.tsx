import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import './index.css';
import {createBrowserRouter, RouterProvider} from 'react-router-dom'
import { AuthCard } from './components/auth/AuthCard.tsx';
const initialView={login:"login",register:"register"}
const router=createBrowserRouter([
  {path:'/',element:<AuthCard initialView={initialView.login} />},
  {path:'/signup',element:<AuthCard initialView={initialView.register} />},
  {path:'/attendance-system',element:<App />},
  {path:'/attendance-system/dashboard',element:<App />},
  {path:'/attendance-system/attendance',element:<App />},
  {path:'/attendance-system/students',element:<App />},
  {path:'/attendance-system/logs',element:<App />},
  {path:'/attendance-system/settings',element:<App />}


])
createRoot(document.getElementById('root')!).render(
  <StrictMode>
     <RouterProvider router={router} />
    {/* <App /> */}
  </StrictMode>
);
