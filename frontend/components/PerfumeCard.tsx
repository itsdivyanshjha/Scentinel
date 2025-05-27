import React from 'react';

type PerfumeCardProps = {
  perfume: {
    _id: string;
    name: string;
    brand: string;
    notes?: string;
    gender?: string;
    description?: string;
    image_url?: string;
    [key: string]: any;
  };
  rank?: number;
  onClick?: () => void;
  isDragging?: boolean;
};

export default function PerfumeCard({ perfume, rank, onClick, isDragging }: PerfumeCardProps) {
  return (
    <div 
      className={`perfume-card ${isDragging ? 'opacity-50' : ''} ${onClick ? 'cursor-pointer' : ''}`}
      onClick={onClick}
    >
      {rank && (
        <div className="absolute -top-3 -left-3 w-8 h-8 bg-primary-600 text-white rounded-full flex items-center justify-center font-bold z-10">
          {rank}
        </div>
      )}
      
      <div className="flex flex-col h-full">
        {perfume.image_url && (
          <div className="mb-4 flex justify-center">
            <img 
              src={perfume.image_url} 
              alt={`${perfume.name} by ${perfume.brand}`}
              className="w-24 h-32 object-contain rounded-md"
              onError={(e) => {
                // Hide image if it fails to load
                e.currentTarget.style.display = 'none';
              }}
            />
          </div>
        )}
        
        <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">{perfume.name}</h3>
        <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">{perfume.brand}</p>
        
        {perfume.gender && (
          <div className="text-xs inline-block px-2 py-1 rounded bg-gray-100 text-gray-600 mb-2 w-fit">
            {perfume.gender.charAt(0).toUpperCase() + perfume.gender.slice(1)}
          </div>
        )}
        
        {perfume.notes && (
          <div className="mt-auto">
            <p className="text-xs font-semibold text-gray-700 mt-2">Notes:</p>
            <p className="text-xs text-gray-600 line-clamp-3">{perfume.notes}</p>
          </div>
        )}
        
        {perfume.description && (
          <div className="mt-2">
            <p className="text-xs font-semibold text-gray-700">Description:</p>
            <p className="text-xs text-gray-600 line-clamp-3">{perfume.description}</p>
          </div>
        )}
      </div>
    </div>
  );
}