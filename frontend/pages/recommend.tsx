import { useState, useEffect } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import axios from 'axios';
import Header from '@/components/Header';
import PerfumeCard from '@/components/PerfumeCard';

type Perfume = {
  _id: string;
  name: string;
  brand: string;
  notes?: string;
  gender?: string;
  description?: string;
  image_url?: string;
  [key: string]: any;
};

export default function RecommendPage() {
  const router = useRouter();
  const [recommendations, setRecommendations] = useState<Perfume[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [warningMessage, setWarningMessage] = useState('');
  const [selectedPerfume, setSelectedPerfume] = useState<Perfume | null>(null);
  const [usePreTrainedModels, setUsePreTrainedModels] = useState(false);

  // Move fetchRecommendations outside useEffect so it can be called from other functions
  const fetchRecommendations = async (skipRankingCheck = false) => {
    setIsLoading(true);
    try {
      const token = localStorage.getItem('token');
      const endpoint = skipRankingCheck 
        ? `${process.env.NEXT_PUBLIC_API_URL}/api/recommend?skip_ranking_check=true` 
        : `${process.env.NEXT_PUBLIC_API_URL}/api/recommend`;
      
      const response = await axios.get(endpoint, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      });
      setRecommendations(response.data);
      setWarningMessage(skipRankingCheck ? 
        'These recommendations are based on our pre-trained models. For more personalized recommendations, please rank some perfumes.' 
        : '');
    } catch (err: any) {
      console.error('Error fetching recommendations:', err);
      if (err.response?.status === 401) {
        // Unauthorized, redirect to login
        localStorage.removeItem('token');
        router.push('/login');
      } else if (err.response?.status === 400) {
        // No rankings found
        setError('You haven\'t ranked any perfumes yet.');
        setUsePreTrainedModels(true); // Show option to use pre-trained models
      } else {
        setError('Failed to load recommendations. Please try again later.');
      }
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    // Fetch recommendations on component mount
    fetchRecommendations();
  }, []);

  const handlePerfumeClick = (perfume: Perfume) => {
    setSelectedPerfume(perfume);
  };

  const closeModal = () => {
    setSelectedPerfume(null);
  };
  
  const handleUsePreTrainedModels = () => {
    setError('');
    setUsePreTrainedModels(false);
    fetchRecommendations(true);
  };
  
  const goToRankPage = () => {
    router.push('/rank');
  };

  return (
    <>
      <Head>
        <title>Your Recommendations - Scentinel</title>
      </Head>
      
      <Header />
      
      <main className="max-w-6xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Your Perfume Recommendations</h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            Based on your preferences, we think you'll love these perfumes.
          </p>
        </div>
        
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-6">
            <p>{error}</p>
            
            {usePreTrainedModels && (
              <div className="mt-4 flex flex-col sm:flex-row space-y-2 sm:space-y-0 sm:space-x-4">
                <button 
                  onClick={goToRankPage}
                  className="btn-primary"
                >
                  Rank Perfumes First
                </button>
                <button 
                  onClick={handleUsePreTrainedModels}
                  className="btn-secondary"
                >
                  Get General Recommendations
                </button>
              </div>
            )}
          </div>
        )}
        
        {warningMessage && (
          <div className="bg-yellow-50 border border-yellow-200 text-yellow-700 px-4 py-3 rounded mb-6">
            {warningMessage}
          </div>
        )}
        
        {isLoading ? (
          <div className="flex justify-center my-12">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-600"></div>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
            {recommendations.map((perfume, index) => (
              <div 
                key={perfume._id} 
                className="relative"
                onClick={() => handlePerfumeClick(perfume)}
              >
                <PerfumeCard 
                  perfume={perfume} 
                  rank={index + 1}
                  onClick={() => handlePerfumeClick(perfume)}
                />
              </div>
            ))}
          </div>
        )}
        
        {recommendations.length === 0 && !isLoading && !error && (
          <div className="flex flex-col items-center justify-center py-12">
            <p className="text-lg text-gray-600 dark:text-gray-400 mb-4">No recommendations available. Try ranking some perfumes first!</p>
            <button 
              onClick={goToRankPage}
              className="btn-primary"
            >
              Go to Ranking Page
            </button>
          </div>
        )}
        
        {/* Detailed perfume modal */}
        {selectedPerfume && (
          <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
            <div className="bg-white dark:bg-gray-800 rounded-lg max-w-2xl w-full max-h-[90vh] overflow-y-auto">
              <div className="p-6">
                <div className="flex justify-between items-start">
                  <div className="flex-1">
                    <h2 className="text-2xl font-bold text-gray-900 dark:text-gray-100">{selectedPerfume.name}</h2>
                    <p className="text-lg text-gray-600 dark:text-gray-400 mt-1">{selectedPerfume.brand}</p>
                  </div>
                  {selectedPerfume.image_url && (
                    <div className="ml-4 flex-shrink-0">
                      <img 
                        src={selectedPerfume.image_url} 
                        alt={`${selectedPerfume.name} by ${selectedPerfume.brand}`}
                        className="w-20 h-28 object-contain rounded-md"
                        onError={(e) => {
                          e.currentTarget.style.display = 'none';
                        }}
                      />
                    </div>
                  )}
                  <button 
                    onClick={closeModal}
                    className="ml-4 text-gray-400 hover:text-gray-500 dark:text-gray-500 dark:hover:text-gray-400"
                  >
                    <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                
                {selectedPerfume.gender && (
                  <div className="mt-4">
                    <span className="text-sm font-semibold text-gray-700 dark:text-gray-300">Gender: </span>
                    <span className="text-sm text-gray-600 dark:text-gray-400">
                      {selectedPerfume.gender.charAt(0).toUpperCase() + selectedPerfume.gender.slice(1)}
                    </span>
                  </div>
                )}
                
                {selectedPerfume.notes && (
                  <div className="mt-4">
                    <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">Notes:</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{selectedPerfume.notes}</p>
                  </div>
                )}
                
                {selectedPerfume.description && (
                  <div className="mt-4">
                    <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300">Description:</h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">{selectedPerfume.description}</p>
                  </div>
                )}
                
                <div className="mt-6 flex justify-end">
                  <button
                    onClick={closeModal}
                    className="btn-secondary"
                  >
                    Close
                  </button>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </>
  );
} 