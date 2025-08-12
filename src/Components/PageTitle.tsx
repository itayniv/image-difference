import React from 'react';

interface PageTitleProps {
  className?: string;
}

const PageTitle: React.FC<PageTitleProps> = ({ className = '' }) => {
  return (
    <div className={`rounded-lg flex flex-col border-gray-200 p-6 mb-6 text-left ${className}`}>
      <h1 className="text-4xl font-bold text-gray-900 mb-2">🖼️</h1>
      <p className="text-lg font-bold text-gray-900 mb-2">
        Image Analyzer
      </p>
      <p className="text-sm text-gray-600 leading-relaxed">
        Pick an AI generated image and a set of reference images to analyze semantic similarity
      </p>
    </div>
  );
};

export default PageTitle;
