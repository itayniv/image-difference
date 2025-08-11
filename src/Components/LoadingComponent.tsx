import React from 'react';

interface LoadingComponentProps {
  /** Progress value between 0 and 1 */
  progress: number;
  /** Optional loading message */
  message?: string;
  /** Optional className for styling */
  className?: string;
  /** Show percentage text */
  showPercentage?: boolean;
  /** Size variant */
  size?: 'small' | 'medium' | 'large';
}

const LoadingComponent: React.FC<LoadingComponentProps> = ({
  progress,
  message = 'Loading...',
  className = '',
  showPercentage = true,
  size = 'medium'
}) => {
  // Clamp progress between 0 and 1
  const clampedProgress = Math.max(0, Math.min(1, progress));
  const percentage = Math.round(clampedProgress * 100);

  const sizeClasses = {
    small: {
      container: 'p-3',
      bar: 'h-2',
      text: 'text-sm'
    },
    medium: {
      container: 'p-4',
      bar: 'h-3',
      text: 'text-base'
    },
    large: {
      container: 'p-6',
      bar: 'h-4',
      text: 'text-lg'
    }
  };

  const currentSize = sizeClasses[size];

  return (
    <div className={`bg-white border border-gray-200 rounded-lg shadow-sm ${currentSize.container} ${className}`}>
      {/* Loading message */}
      <div className={`text-gray-700 mb-3 font-medium ${currentSize.text}`}>
        {message}
      </div>

      {/* Progress bar container */}
      <div className={`w-full bg-gray-200 rounded-full overflow-hidden ${currentSize.bar}`}>
        {/* Progress bar fill */}
        <div
          className={`bg-gradient-to-r from-blue-500 to-blue-600 ${currentSize.bar} rounded-full transition-all duration-300 ease-out`}
          style={{ width: `${percentage}%` }}
        />
      </div>

      {/* Percentage and progress indicator */}
      <div className="flex justify-between items-center mt-2">
        {showPercentage && (
          <span className={`text-gray-600 font-mono ${currentSize.text === 'text-lg' ? 'text-base' : 'text-sm'}`}>
            {percentage}%
          </span>
        )}
        <div className="flex space-x-1">
          {/* Animated dots */}
          {[0, 1, 2].map((i) => (
            <div
              key={i}
              className={`w-1.5 h-1.5 bg-blue-500 rounded-full animate-pulse`}
              style={{
                animationDelay: `${i * 0.2}s`,
                animationDuration: '1s'
              }}
            />
          ))}
        </div>
      </div>

      {/* Progress details for debugging if needed */}
      {process.env.NODE_ENV === 'development' && (
        <div className="mt-2 text-xs text-gray-400 font-mono">
          Raw progress: {progress.toFixed(4)}
        </div>
      )}
    </div>
  );
};

export default LoadingComponent;
