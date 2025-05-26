import { useState, useEffect } from 'react';
import Head from 'next/head';
import { useRouter } from 'next/router';
import axios from 'axios';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';
import Header from '@/components/Header';
import PerfumeCard from '@/components/PerfumeCard';

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

  useEffect(() => {
    // Fetch perfumes to rank
    const fetchPerfumes = async () => {
      setIsLoading(true);
      try {
        const token = localStorage.getItem('token');
        const response = await axios.get(`${process.env.NEXT_PUBLIC_API_URL}/api/perfumes/list`, {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        });
        setPerfumes(response.data);
      } catch (err: any) {
        console.error('Error fetching perfumes:', err);
        if (err.response?.status === 401) {
          // Unauthorized, redirect to login
          localStorage.removeItem('token');
          router.push('/login');
        } else {
          setError('Failed to load perfumes. Please try again later.');
        }
      } finally {
        setIsLoading(false);
      }
    };

    fetchPerfumes();
  }, []);

  const handleDragEnd = (result: any) => {
    if (!result.destination) return;

    const items = Array.from(perfumes);
    const [reorderedItem] = items.splice(result.source.index, 1);
    items.splice(result.destination.index, 0, reorderedItem);

    setPerfumes(items);
  };

  const handleSubmitRankings = async () => {
    setIsSubmitting(true);
    setError('');
    
    try {
      const token = localStorage.getItem('token');
      const rankings = perfumes.map((perfume, index) => ({
        perfume_id: perfume._id,
        rank: index + 1, // 1-based ranking (1 is highest)
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
      
      // Redirect to recommendations after 2 seconds
      setTimeout(() => {
        router.push('/recommend');
      }, 2000);
      
    } catch (err: any) {
      console.error('Error submitting rankings:', err);
      if (err.response?.status === 401) {
        // Unauthorized, redirect to login
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
      
      <main className="max-w-6xl mx-auto py-8 px-4 sm:px-6 lg:px-8">
        <div className="mb-8">
          <h1 className="text-3xl font-bold text-gray-900">Rank Your Perfumes</h1>
          <p className="mt-2 text-gray-600">
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
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-600"></div>
          </div>
        ) : (
          <>
            <DragDropContext onDragEnd={handleDragEnd}>
              <Droppable droppableId="perfumes" direction="vertical">
                {(provided) => (
                  <div
                    {...provided.droppableProps}
                    ref={provided.innerRef}
                    className="space-y-4"
                  >
                    {perfumes.map((perfume, index) => (
                      <Draggable key={perfume._id} draggableId={perfume._id} index={index}>
                        {(provided, snapshot) => (
                          <div
                            ref={provided.innerRef}
                            {...provided.draggableProps}
                            {...provided.dragHandleProps}
                            className="relative"
                          >
                            <div className="flex items-center">
                              <div className="mr-4 w-8 h-8 bg-primary-600 text-white rounded-full flex items-center justify-center font-bold">
                                {index + 1}
                              </div>
                              <div className="flex-1">
                                <PerfumeCard 
                                  perfume={perfume} 
                                  isDragging={snapshot.isDragging} 
                                />
                              </div>
                              <div className="ml-4 text-gray-400">
                                <svg xmlns="http://www.w3.org/2000/svg" className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                                </svg>
                              </div>
                            </div>
                          </div>
                        )}
                      </Draggable>
                    ))}
                    {provided.placeholder}
                  </div>
                )}
              </Droppable>
            </DragDropContext>
            
            <div className="mt-8 flex justify-center">
              <button
                onClick={handleSubmitRankings}
                disabled={isSubmitting || submitSuccess}
                className="btn-primary px-6 py-3 flex items-center space-x-2"
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