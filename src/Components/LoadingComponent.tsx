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
    <div className={`bg-white flex flex-row items-center justify-center rounded-lg shadow-sm ${currentSize.container} ${className}`}>
      {/* Message and progress bar side by side */}
      <div className="flex flex-row items-center w-full gap-6">
        {/* Loading message - now with flex-shrink-0 to prevent wrapping */}
        <div className={`text-gray-700 flex-shrink-0 font-medium ${currentSize.text}`}>
          {message}
        </div>

        {/* Progress bar container - now takes remaining space */}
        <div className="flex-1 flex flex-col w-full">
          {/* Progress bar */}
          <div className={`w-full bg-gray-200 rounded-full overflow-hidden ${currentSize.bar}`}>
            {/* Progress bar fill */}
            <div
              className={`bg-gradient-to-r from-blue-500 to-blue-600 ${currentSize.bar} rounded-full transition-all duration-300 ease-out`}
              style={{ width: `${percentage}%` }}
            />
          </div>

          {/* Percentage indicator below progress bar */}
          {showPercentage && (
            <div className="flex justify-end mt-1">
              <span className={`text-gray-600 font-mono ${currentSize.text === 'text-lg' ? 'text-base' : 'text-sm'}`}>
                {progress.toFixed(0)}%
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default LoadingComponent;
