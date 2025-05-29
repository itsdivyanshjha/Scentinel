import { useState, useEffect } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import axios from 'axios';
import Header from '@/components/Header';

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

  const getGenderColor = (gender?: string) => {
    if (!gender) return 'bg-gray-100 text-gray-600';
    const g = gender.toLowerCase();
    if (g.includes('women') || g.includes('female')) return 'bg-pink-100 text-pink-700';
    if (g.includes('men') || g.includes('male')) return 'bg-blue-100 text-blue-700';
    return 'bg-purple-100 text-purple-700'; // For unisex or others
  };

  const getGenderIcon = (gender?: string) => {
    if (!gender) return '👤';
    const g = gender.toLowerCase();
    if (g.includes('women') || g.includes('female')) return '👩';
    if (g.includes('men') || g.includes('male')) return '👨';
    return '👥'; // For unisex
  };

  return (
    <>
      <Head>
        <title>Your Recommendations - Scentinel</title>
      </Head>
      
      <Header />
      
      <main className="min-h-screen bg-gradient-to-br from-gray-50 to-gray-100 dark:from-gray-900 dark:to-gray-800">
        <div className="max-w-7xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
          {/* Hero Section */}
          <div className="text-center mb-12">
            <h1 className="text-4xl font-bold text-gray-900 dark:text-gray-100 mb-4">
              ✨ Your Personalized Fragrance Journey
            </h1>
            <p className="text-xl text-gray-600 dark:text-gray-400 max-w-2xl mx-auto">
              Discover perfumes crafted for your unique taste profile through our AI-powered recommendation engine
            </p>
          </div>
          
          {error && (
            <div className="bg-red-50 border-l-4 border-red-400 text-red-700 p-6 rounded-lg mb-8 shadow-sm">
              <div className="flex items-center">
                <svg className="w-6 h-6 mr-3 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7-4a1 1 0 11-2 0 1 1 0 012 0zM9 9a1 1 0 000 2v3a1 1 0 001 1h1a1 1 0 100-2v-3a1 1 0 00-1-1H9z" clipRule="evenodd" />
                </svg>
                <p className="font-medium">{error}</p>
              </div>
              
              {usePreTrainedModels && (
                <div className="mt-6 flex flex-col sm:flex-row space-y-3 sm:space-y-0 sm:space-x-4">
                  <button 
                    onClick={goToRankPage}
                    className="inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-lg shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-colors"
                  >
                    🎯 Rank Perfumes First
                  </button>
                  <button 
                    onClick={handleUsePreTrainedModels}
                    className="inline-flex items-center px-6 py-3 border border-gray-300 text-base font-medium rounded-lg text-gray-700 bg-white hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-colors"
                  >
                    🤖 Get General Recommendations
                  </button>
                </div>
              )}
            </div>
          )}
          
          {warningMessage && (
            <div className="bg-amber-50 border-l-4 border-amber-400 text-amber-700 p-6 rounded-lg mb-8 shadow-sm">
              <div className="flex items-center">
                <svg className="w-6 h-6 mr-3 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                  <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                </svg>
                <p className="font-medium">{warningMessage}</p>
              </div>
            </div>
          )}
          
          {isLoading ? (
            <div className="flex flex-col items-center justify-center py-20">
              <div className="relative">
                <div className="animate-spin rounded-full h-16 w-16 border-t-4 border-b-4 border-primary-600"></div>
                <div className="absolute inset-0 flex items-center justify-center">
                  <span className="text-2xl">🌸</span>
                </div>
              </div>
              <p className="mt-4 text-lg text-gray-600 dark:text-gray-400 font-medium">
                Crafting your perfect fragrance matches...
              </p>
            </div>
          ) : (
            <>
              {recommendations.length > 0 && (
                <div className="text-center mb-8">
                  <div className="inline-flex items-center px-4 py-2 bg-primary-100 text-primary-800 rounded-full text-sm font-medium">
                    🎯 {recommendations.length} Personalized Recommendations
                  </div>
                </div>
              )}
              
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-8">
                {recommendations.map((perfume, index) => (
                  <div 
                    key={perfume._id} 
                    className="group relative bg-white dark:bg-gray-800 rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-2 cursor-pointer overflow-hidden mt-4 ml-4"
                    onClick={() => handlePerfumeClick(perfume)}
                  >
                    {/* Rank Badge */}
                    <div className="absolute -top-2 -left-2 w-10 h-10 bg-gradient-to-r from-primary-600 to-primary-700 text-white rounded-full flex items-center justify-center font-bold z-10 shadow-lg text-sm">
                      #{index + 1}
                    </div>
                    
                    {/* Gender Badge */}
                    {perfume.gender && (
                      <div className={`absolute top-4 right-4 px-3 py-1 rounded-full text-xs font-semibold ${getGenderColor(perfume.gender)} backdrop-blur-sm`}>
                        <span className="mr-1">{getGenderIcon(perfume.gender)}</span>
                        {perfume.gender.charAt(0).toUpperCase() + perfume.gender.slice(1)}
                      </div>
                    )}
                    
                    <div className="p-6">
                      {/* Perfume Image */}
                      {perfume.image_url ? (
                        <div className="mb-6 flex justify-center">
                          <div className="relative">
                            <img 
                              src={perfume.image_url} 
                              alt={`${perfume.name} by ${perfume.brand}`}
                              className="w-20 h-28 object-contain rounded-lg shadow-md group-hover:scale-105 transition-transform duration-300"
                              onError={(e) => {
                                e.currentTarget.style.display = 'none';
                              }}
                            />
                            <div className="absolute inset-0 bg-gradient-to-t from-black/10 to-transparent rounded-lg"></div>
                          </div>
                        </div>
                      ) : (
                        <div className="mb-6 flex justify-center">
                          <div className="w-20 h-28 bg-gradient-to-br from-primary-100 to-primary-200 rounded-lg flex items-center justify-center">
                            <span className="text-3xl">🌸</span>
                          </div>
                        </div>
                      )}
                      
                      {/* Perfume Details */}
                      <div className="text-center mb-4">
                        <h3 className="text-lg font-bold text-gray-900 dark:text-gray-100 mb-1 line-clamp-2">
                          {perfume.name}
                        </h3>
                        <p className="text-sm text-gray-600 dark:text-gray-400 font-medium">
                          {perfume.brand}
                        </p>
                      </div>
                      
                      {/* Notes Preview */}
                      {perfume.notes && (
                        <div className="mb-4">
                          <p className="text-xs font-semibold text-gray-700 dark:text-gray-300 mb-1">
                            🌿 Fragrance Notes:
                          </p>
                          <p className="text-xs text-gray-600 dark:text-gray-400 line-clamp-2 leading-relaxed">
                            {perfume.notes}
                          </p>
                        </div>
                      )}
                      
                      {/* Description Preview */}
                      {perfume.description && (
                        <div className="mb-4">
                          <p className="text-xs text-gray-500 dark:text-gray-400 line-clamp-2 italic">
                            "{perfume.description}"
                          </p>
                        </div>
                      )}
                      
                      {/* Click to view more */}
                      <div className="text-center">
                        <span className="text-xs text-primary-600 dark:text-primary-400 font-medium group-hover:text-primary-700 transition-colors">
                          Click to view details →
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
          
          {recommendations.length === 0 && !isLoading && !error && (
            <div className="flex flex-col items-center justify-center py-20">
              <div className="text-6xl mb-6">🤔</div>
              <h3 className="text-2xl font-bold text-gray-900 dark:text-gray-100 mb-4">
                No recommendations yet
              </h3>
              <p className="text-lg text-gray-600 dark:text-gray-400 mb-8 text-center max-w-md">
                Let us learn about your fragrance preferences to create personalized recommendations just for you!
              </p>
              <button 
                onClick={goToRankPage}
                className="inline-flex items-center px-8 py-4 border border-transparent text-lg font-medium rounded-xl shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500 transition-colors"
              >
                🎯 Start Ranking Perfumes
              </button>
            </div>
          )}
          
          {/* Enhanced Detailed perfume modal */}
          {selectedPerfume && (
            <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center p-4 z-50 backdrop-blur-sm">
              <div className="bg-white dark:bg-gray-800 rounded-2xl max-w-4xl w-full max-h-[90vh] overflow-y-auto shadow-2xl">
                <div className="relative">
                  {/* Header with gradient */}
                  <div className="bg-gradient-to-r from-primary-600 to-primary-700 p-6 rounded-t-2xl">
                    <div className="flex justify-between items-start">
                      <div className="flex-1 text-white">
                        <h2 className="text-3xl font-bold mb-2">{selectedPerfume.name}</h2>
                        <p className="text-xl opacity-90">{selectedPerfume.brand}</p>
                        {selectedPerfume.gender && (
                          <div className="mt-3 inline-flex items-center px-3 py-1 bg-white/20 rounded-full text-sm font-medium">
                            <span className="mr-2">{getGenderIcon(selectedPerfume.gender)}</span>
                            {selectedPerfume.gender.charAt(0).toUpperCase() + selectedPerfume.gender.slice(1)}
                          </div>
                        )}
                      </div>
                      
                      {selectedPerfume.image_url && (
                        <div className="ml-6 flex-shrink-0">
                          <img 
                            src={selectedPerfume.image_url} 
                            alt={`${selectedPerfume.name} by ${selectedPerfume.brand}`}
                            className="w-24 h-32 object-contain rounded-lg shadow-lg bg-white/10 backdrop-blur-sm p-2"
                            onError={(e) => {
                              e.currentTarget.style.display = 'none';
                            }}
                          />
                        </div>
                      )}
                      
                      <button 
                        onClick={closeModal}
                        className="ml-4 p-2 text-white/80 hover:text-white hover:bg-white/20 rounded-lg transition-all"
                      >
                        <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                      </button>
                    </div>
                  </div>
                  
                  {/* Content */}
                  <div className="p-6 space-y-6">
                    {selectedPerfume.notes && (
                      <div className="bg-gray-50 dark:bg-gray-700 rounded-xl p-4">
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3 flex items-center">
                          🌿 Fragrance Notes
                        </h3>
                        <p className="text-gray-700 dark:text-gray-300 leading-relaxed">
                          {selectedPerfume.notes}
                        </p>
                      </div>
                    )}
                    
                    {selectedPerfume.description && (
                      <div className="bg-primary-50 dark:bg-primary-900/20 rounded-xl p-4">
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3 flex items-center">
                          📝 Description
                        </h3>
                        <p className="text-gray-700 dark:text-gray-300 leading-relaxed italic">
                          "{selectedPerfume.description}"
                        </p>
                      </div>
                    )}
                    
                    {/* Additional metadata if available */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {Object.entries(selectedPerfume).map(([key, value]) => {
                        if (['_id', 'name', 'brand', 'notes', 'description', 'image_url', 'gender'].includes(key) || !value) return null;
                        return (
                          <div key={key} className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3">
                            <span className="text-sm font-semibold text-gray-700 dark:text-gray-300 capitalize">
                              {key.replace(/_/g, ' ')}:
                            </span>
                            <span className="text-sm text-gray-600 dark:text-gray-400 ml-2">
                              {String(value)}
                            </span>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                  
                  {/* Footer */}
                  <div className="border-t border-gray-200 dark:border-gray-700 p-6">
                    <div className="flex justify-end">
                      <button
                        onClick={closeModal}
                        className="px-6 py-3 border border-gray-300 dark:border-gray-600 text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 hover:bg-gray-50 dark:hover:bg-gray-600 rounded-lg font-medium transition-colors"
                      >
                        Close
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </>
  );
} 