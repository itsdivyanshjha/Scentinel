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
  [key: string]: any;
};

export default function RankPage() {
  const router = useRouter();
  const [perfumes, setPerfumes] = useState<Perfume[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitSuccess, setSubmitSuccess] = useState(false);
  const [draggedIndex, setDraggedIndex] = useState<number | null>(null);
  const [dragOverIndex, setDragOverIndex] = useState<number | null>(null);

  // Helper function to extract ObjectId string
  const getObjectIdString = (id: any): string => {
    if (typeof id === 'string') return id;
    if (id && id.$oid) return id.$oid;
    return String(id);
  };

  useEffect(() => {
    const fetchPerfumes = async () => {
      setIsLoading(true);
      
      try {
        const token = localStorage.getItem('token');
        
        if (!token) {
          setError('Please log in to rank perfumes.');
          router.push('/login');
          return;
        }

        const response = await axios.get(`${process.env.NEXT_PUBLIC_API_URL}/api/perfumes/list`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        
        if (response.data && Array.isArray(response.data)) {
          setPerfumes(response.data);
        } else {
          setError('Invalid response format from server');
        }
      } catch (err: any) {
        console.error('Error fetching perfumes:', err);
        
        if (err.response?.status === 401) {
          localStorage.removeItem('token');
          router.push('/login');
        } else {
          setError(`Failed to load perfumes: ${err.response?.data?.error || err.message}`);
        }
      } finally {
        setIsLoading(false);
      }
    };

    fetchPerfumes();
  }, [router]);

  const handleDragStart = (e: React.DragEvent, index: number) => {
    e.dataTransfer.setData('text/plain', index.toString());
    setDraggedIndex(index);
  };

  const handleDragOver = (e: React.DragEvent, index: number) => {
    e.preventDefault();
    setDragOverIndex(index);
  };

  const handleDragLeave = () => {
    setDragOverIndex(null);
  };

  const handleDrop = (e: React.DragEvent, dropIndex: number) => {
    e.preventDefault();
    const dragIndex = parseInt(e.dataTransfer.getData('text/plain'));
    
    if (dragIndex !== dropIndex) {
      const items = Array.from(perfumes);
      const [reorderedItem] = items.splice(dragIndex, 1);
      items.splice(dropIndex, 0, reorderedItem);
      setPerfumes(items);
    }
    setDraggedIndex(null);
    setDragOverIndex(null);
  };

  const handleDragEnd = () => {
    setDraggedIndex(null);
    setDragOverIndex(null);
  };

  const handleSubmitRankings = async () => {
    setIsSubmitting(true);
    setError('');
    
    try {
      const token = localStorage.getItem('token');
      const rankings = perfumes.map((perfume, index) => ({
        perfume_id: getObjectIdString(perfume._id),
        rank: index + 1,
      }));
      
      await axios.post(
        `${process.env.NEXT_PUBLIC_API_URL}/api/perfumes/rank`,
        { rankings },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );
      
      setSubmitSuccess(true);
      setTimeout(() => {
        router.push('/recommend');
      }, 2000);
      
    } catch (err: any) {
      console.error('Error submitting rankings:', err);
      if (err.response?.status === 401) {
        localStorage.removeItem('token');
        router.push('/login');
      } else {
        setError('Failed to submit rankings. Please try again later.');
      }
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <>
      <Head>
        <title>Rank Perfumes - Scentinel</title>
      </Head>
      
      <Header />
      
      <main className="max-w-4xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">Rank Your Perfumes</h1>
          <p className="mt-2 text-gray-600 dark:text-gray-400">
            Drag and drop the perfumes to rank them from 1 (most liked) to 10 (least liked).
          </p>
        </div>
        
        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded mb-6">
            {error}
          </div>
        )}
        
        {submitSuccess && (
          <div className="bg-green-50 border border-green-200 text-green-700 px-4 py-3 rounded mb-6">
            Rankings submitted successfully! Redirecting to recommendations...
          </div>
        )}
        
        {isLoading ? (
          <div className="flex justify-center my-12">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-blue-600"></div>
          </div>
        ) : perfumes.length === 0 ? (
          <div className="text-center py-12">
            <p className="text-gray-500 dark:text-gray-400">No perfumes available for ranking.</p>
            <button 
              onClick={() => window.location.reload()} 
              className="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700"
            >
              Retry
            </button>
          </div>
        ) : (
          <>
            <div className="space-y-3">
              {perfumes.map((perfume, index) => (
                <div
                  key={getObjectIdString(perfume._id)}
                  className={`
                    flex items-center p-4 bg-white rounded-lg border-2 transition-all duration-200 cursor-move
                    ${draggedIndex === index ? 'border-blue-400 shadow-lg scale-105 opacity-50' : 
                      dragOverIndex === index ? 'border-green-400 bg-green-50' :
                      'border-gray-200 hover:border-gray-300 hover:shadow-md'}
                  `}
                  draggable={true}
                  onDragStart={(e) => handleDragStart(e, index)}
                  onDragOver={(e) => handleDragOver(e, index)}
                  onDragLeave={handleDragLeave}
                  onDrop={(e) => handleDrop(e, index)}
                  onDragEnd={handleDragEnd}
                >
                  {/* Rank Number */}
                  <div className="flex-shrink-0 w-10 h-10 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold text-lg mr-4">
                    {index + 1}
                  </div>
                  
                  {/* Perfume Info */}
                  <div className="flex-1 min-w-0">
                    <h3 className="text-lg font-semibold text-gray-900 truncate">
                      {perfume.name || 'Unknown Perfume'}
                    </h3>
                    <p className="text-sm text-gray-600 mb-1">
                      {perfume.brand || 'Unknown Brand'}
                    </p>
                    
                    {perfume.gender && (
                      <span className="inline-block px-2 py-1 text-xs bg-gray-100 text-gray-600 rounded-full">
                        {perfume.gender.charAt(0).toUpperCase() + perfume.gender.slice(1)}
                      </span>
                    )}
                    
                    {perfume.notes && (
                      <p className="text-sm text-gray-500 mt-2 line-clamp-2">
                        <span className="font-medium">Notes:</span> {perfume.notes}
                      </p>
                    )}
                  </div>
                  
                  {/* Drag Handle */}
                  <div className="flex-shrink-0 ml-4 text-gray-400 hover:text-gray-600 cursor-grab active:cursor-grabbing">
                    <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                    </svg>
                  </div>
                </div>
              ))}
            </div>
            
            <div className="mt-8 flex justify-center">
              <button
                onClick={handleSubmitRankings}
                disabled={isSubmitting || submitSuccess}
                className="bg-blue-600 text-white px-8 py-3 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
              >
                {isSubmitting ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-t-2 border-b-2 border-white"></div>
                    <span>Submitting...</span>
                  </>
                ) : (
                  <span>Submit Rankings</span>
                )}
              </button>
            </div>
          </>
        )}
      </main>
    </>
  );
} 