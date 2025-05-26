import { useEffect } from 'react';
import { useRouter } from 'next/router';
import Head from 'next/head';

export default function Home() {
  const router = useRouter();

  useEffect(() => {
    // Redirect to rank page
    router.push('/rank');
  }, []);

  return (
    <>
      <Head>
        <title>Scentinel - Perfume Recommendation System</title>
        <meta name="description" content="Find your perfect perfume match" />
      </Head>
      <main className="flex min-h-screen flex-col items-center justify-center p-24">
        <div className="flex flex-col items-center justify-center">
          <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-primary-600"></div>
          <h2 className="mt-6 text-2xl font-semibold">Loading...</h2>
        </div>
      </main>
    </>
  );
} 