'use client';

import React from 'react';
import { Input } from './input';
import { Button } from './button';
import { Check, X } from 'lucide-react';

interface InputLabelProps {
  value: string;
  isEditing: boolean;
  onEdit: () => void;
  onSave: (value: string) => void;
  onCancel: () => void;
  onChange: (value: string) => void;
  className?: string;
}

export function InputLabel({
  value,
  isEditing,
  onEdit,
  onSave,
  onCancel,
  onChange,
  className
}: InputLabelProps) {
  const [tempValue, setTempValue] = React.useState(value);

  React.useEffect(() => {
    setTempValue(value);
  }, [value]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      onSave(tempValue);
    } else if (e.key === 'Escape') {
      onCancel();
    }
  };

  if (isEditing) {
    return (
      <div className="flex items-center gap-1">
        <Input
          value={tempValue}
          onChange={(e) => {
            setTempValue(e.target.value);
            onChange(e.target.value);
          }}
          onKeyDown={handleKeyDown}
          className="h-6 py-1 px-2 text-sm"
          autoFocus
        />
        <div className="flex gap-1">
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={() => onSave(tempValue)}
          >
            <Check className="h-3 w-3" />
          </Button>
          <Button
            variant="ghost"
            size="icon"
            className="h-6 w-6"
            onClick={onCancel}
          >
            <X className="h-3 w-3" />
          </Button>
        </div>
      </div>
    );
  }

  return (
    <span
      onClick={onEdit}
      className={className}
    >
      {value}
    </span>
  );
}
