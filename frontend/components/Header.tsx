import { useRouter } from 'next/router';
import Link from 'next/link';

export default function Header() {
  const router = useRouter();

  const handleLogout = () => {
    // Clear localStorage
    localStorage.removeItem('token');
    localStorage.removeItem('userId');
    
    // Redirect to login
    router.push('/login');
  };

  return (
    <header className="bg-white shadow-sm">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between h-16">
          <div className="flex">
            <div className="flex-shrink-0 flex items-center">
              <Link href="/" className="text-2xl font-bold text-primary-700">
                Scentinel
              </Link>
            </div>
            <nav className="ml-6 flex items-center space-x-4">
              <Link 
                href="/rank" 
                className={`px-3 py-2 rounded-md text-sm font-medium ${
                  router.pathname === '/rank' 
                    ? 'text-primary-700 bg-primary-50'
                    : 'text-gray-600 hover:text-primary-600 hover:bg-gray-50'
                }`}
              >
                Rank Perfumes
              </Link>
              <Link 
                href="/recommend" 
                className={`px-3 py-2 rounded-md text-sm font-medium ${
                  router.pathname === '/recommend' 
                    ? 'text-primary-700 bg-primary-50'
                    : 'text-gray-600 hover:text-primary-600 hover:bg-gray-50'
                }`}
              >
                Recommendations
              </Link>
            </nav>
          </div>
          <div className="flex items-center">
            <button
              onClick={handleLogout}
              className="px-3 py-2 rounded-md text-sm font-medium text-gray-600 hover:text-primary-600 hover:bg-gray-50"
            >
              Logout
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}